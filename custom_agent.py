import os
import re
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
import diskcache as dc

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.schema import StrOutputParser, AIMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai


# ---------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="What's up Doc? 🤖🩺")
st.title("What's up Doc? ⚕️📖")

# ---------------------------------------------------------------------
# Rate limiting setup
# ---------------------------------------------------------------------
RATE_LIMIT_SECONDS = 60
MAX_TOKENS_PER_MINUTE = 10_000_000
request_tokens = []


def is_rate_limited(tokens_this_request: int) -> bool:
    """Token-based rate limiter."""
    global request_tokens
    now = time.time()
    request_tokens = [(t, tok) for (t, tok) in request_tokens if now - t < RATE_LIMIT_SECONDS]
    tokens_used = sum(tok for _, tok in request_tokens)
    if tokens_used + tokens_this_request > MAX_TOKENS_PER_MINUTE:
        return True
    request_tokens.append((now, tokens_this_request))
    return False


# ---------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------
with st.sidebar:
    k_value = st.number_input("K value", min_value=1, max_value=10, value=3)
    use_secrets = st.toggle("Use Streamlit Secrets for API Key", value=True)

    if use_secrets:
        try:
            gemini_api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
            st.stop()
    else:
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key!", icon="⚠")
            st.stop()

os.environ["GOOGLE_API_KEY"] = gemini_api_key
genai.configure(api_key=gemini_api_key)

# ---------------------------------------------------------------------
# Memory and cache
# ---------------------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3)

cache = dc.Cache("medical_cache")

# ---------------------------------------------------------------------
# Search tool with caching + transparency
# ---------------------------------------------------------------------
search_engine = DuckDuckGoSearchRun()


def medical_search(query: str) -> Dict[str, Any]:
    """Cached multi-source medical search with transparency messages."""
    query_key = query.strip().lower()

    if query_key in cache:
        data = cache[query_key]
        st.info(f"🔁 Using cached results for '{query}' (last updated {data['timestamp']}).")
        return data["results"]

    st.info(f"🌐 Searching live sources for '{query}' ...")

    sources = [
        "webmd.com",
        "mayoclinic.org",
        "nih.gov",
        "cdc.gov",
        "clevelandclinic.org",
    ]

    results = {}
    for src in sources:
        try:
            res = search_engine.run(f"site:{src} {query}")
            if res:
                results[src] = res[:400]
        except Exception as e:
            results[src] = f"Search failed ({e})"

    cache[query_key] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    cache.expire(query_key, 60 * 60 * 24 * 7)  # expire after 7 days
    return results


medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites for evidence-based information.",
)

# ---------------------------------------------------------------------
# Summarisation prompt for multiple sources
# ---------------------------------------------------------------------
summarise_prompt = ChatPromptTemplate.from_template("""
You have collected factual information from several reliable medical websites.

Summarise their content **objectively**:

- Use **Markdown bullets** for each key point, such as medications, treatments, or symptoms.
- Include the **source** for each bullet in parentheses.
- Highlight areas of **agreement** and **disagreement**.
- Keep it concise and readable.
- End by reminding users to consult a healthcare professional.

**Sources provided:**  
{sources}

**User question:**  
{question}

Format your answer entirely in Markdown.
""")


# ---------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)

summarise_chain = summarise_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------
# Main medical conversation prompt
# ---------------------------------------------------------------------
medical_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly, evidence-based AI medical assistant.
Always be clear, factual, and compassionate.
Never give drug dosages or replace professional diagnosis.
Encourage seeing a healthcare provider for serious or uncertain cases."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ---------------------------------------------------------------------
# Smart routing: when to search vs. direct reasoning
# ---------------------------------------------------------------------
def should_search(input_text: str) -> bool:
    keywords = [
        "treat", "symptom", "drug", "drugs", "medication", "medications",
        "cause", "prevent", "diagnose", "test", "therapy", "dose", "prescribe"
    ]
    input_text_lower = input_text.lower()
    return any(word in input_text_lower for word in keywords)


def route(input_text: str):
    return "search" if should_search(input_text) else "no_search"


def format_sources(src_dict):
    formatted = ""
    for site, content in src_dict.items():
        # Split content into sentences using simple punctuation
        sentences = re.split(r'(?<=[.!?]) +', content)
        snippet = " ".join(sentences[:5])  # Take up to 5 sentences
        formatted += f"- **{site}**:\n  {snippet}\n\n"
    return formatted


# --- define search branch safely ---
search_branch = (
    # Step 1: Perform cached medical search
    RunnableLambda(lambda x: {
        "results": medical_search(x["input"]),
        "question": x["input"],
        "history": x.get("history", [])
    })

    # Step 2: Format results for summarisation
    | RunnableLambda(lambda x: {
        "sources": format_sources(x["results"]),
        "question": x["question"],
        "history": x["history"]
    })

    # Step 3: Summarise sources with safe fallback
    | RunnableLambda(lambda x: {
        "summary": summarise_chain.invoke({
            "sources": x["sources"],
            "question": x["question"]
        }) if x["sources"].strip() else "No verified sources found for this query.",
        "question": x["question"],
        "history": x["history"]
    })

    # Step 4: Prepare input for final LLM
    | RunnableLambda(lambda x: {
        "input": f"**Question:** {x['question']}\n\n**Verified info:**\n{x['summary']}",
        "history": x["history"]
    })

    # Step 5: Generate user-facing answer
    | medical_prompt
    | llm
    | StrOutputParser()
)



no_search_branch = medical_prompt | llm | StrOutputParser()

# Corrected RunnableBranch syntax
router_chain = RunnableBranch(
    (lambda x: route(x["input"]) == "search", search_branch),  
    no_search_branch                                    
)

# ---------------------------------------------------------------------
# Response generation helper
# ---------------------------------------------------------------------
def get_medical_answer(query: str) -> str:
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit before sending more questions."

    context = {"input": query, "history": st.session_state.memory.chat_memory.messages}
    response = router_chain.invoke(context)
    return response.strip()

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("Ask your medical question")
    submit = st.form_submit_button("Submit")

if submit:
    gif_runner = st.image("doc.gif", caption="Processing...", output_format="auto")
    answer = get_medical_answer(user_query)
    time.sleep(0.6)
    gif_runner.empty()

    st.markdown("### 💬 Suggestion")
    # Convert line breaks into Markdown-friendly format
    formatted_answer = answer.replace("\n", "  \n")
    st.markdown(formatted_answer, unsafe_allow_html=True)


    st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_query))
    st.session_state.memory.chat_memory.add_message(AIMessage(content=answer))

# Display chat history in a collapsible expander
if st.session_state.memory.chat_memory.messages:
    st.markdown("---")
    with st.expander("🩺 View Chat History", expanded=False):
        history_md = ""
        for m in st.session_state.memory.chat_memory.messages[-10:]:
            if isinstance(m, HumanMessage):
                history_md += f"**You:** {m.content}  \n"
            else:
                content = m.content.replace("\n", "  \n")
                history_md += f"**DocBot:**  \n{content}  \n\n"
        st.markdown(history_md, unsafe_allow_html=True)


# Sidebar cache management
with st.sidebar:
    if st.button("Clear cache"):
        cache.clear()
        st.success("Cache cleared.")
