import os
import re
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
import diskcache as dc

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableSequence
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.schema import StrOutputParser, AIMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="What's up Doc? 🤖🩺")
st.title("What's up Doc? ⚕️📖")

# -----------------------
# Rate limiting
# -----------------------
RATE_LIMIT_SECONDS = 60
MAX_TOKENS_PER_MINUTE = 10_000_000
request_tokens = []

def is_rate_limited(tokens_this_request: int) -> bool:
    global request_tokens
    now = time.time()
    # Remove old requests outside window
    request_tokens = [(t, tok) for (t, tok) in request_tokens if now - t < RATE_LIMIT_SECONDS]
    tokens_used = sum(tok for _, tok in request_tokens)
    if tokens_used + tokens_this_request > MAX_TOKENS_PER_MINUTE:
        return True
    request_tokens.append((now, tokens_this_request))
    return False

# -----------------------
# Memory and cache
# -----------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3)

cache = dc.Cache("medical_cache")  # define cache first

# -----------------------
# Sidebar with cache button
# -----------------------
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

    # Clear cache button in sidebar
    if st.button("Clear Cache"):
        cache.clear()
        st.success("✅ Cache cleared!")

os.environ["GOOGLE_API_KEY"] = gemini_api_key
genai.configure(api_key=gemini_api_key)

# -----------------------
# Search tool with caching
# -----------------------
search_engine = DuckDuckGoSearchRun()

def medical_search(query: str) -> Dict[str, Any]:
    """Cached multi-source medical search with safe sources."""
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
                results[src] = res[:400]  # truncate to avoid too long text
        except Exception as e:
            results[src] = f"Search failed ({e})"

    cache[query_key] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    cache.expire(query_key, 60 * 60 * 24 * 7)
    return results

medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites for evidence-based information."
)

# -----------------------
# Summarisation prompt
# -----------------------
summarise_prompt = ChatPromptTemplate.from_template("""
You have collected factual information from several reliable medical websites.

Summarise their content **objectively**:

- Use **Markdown bullets** for key points (medications, treatments, symptoms).
- Include **source** in parentheses **for every fact**.
- Highlight areas of **agreement** and **disagreement**.
- Keep it concise and readable.
- End by reminding users to consult a healthcare professional.

Sources (include these in your answer):
{sources}

User question:
{question}

Format your answer entirely in Markdown.
""")

# Summarisation chain replacement
summarise_runnable = (
    summarise_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)

# -----------------------
# Routing logic
# -----------------------
def should_search(input_text: str) -> bool:
    keywords = [
        "treat", "symptom", "drug", "drugs", "medication", "medications",
        "cause", "prevent", "diagnose", "test", "therapy", "dose", "prescribe"
    ]
    return any(word in input_text.lower() for word in keywords)

def route(input_text: str):
    return "search" if should_search(input_text) else "no_search"

def format_sources(src_dict: dict) -> str:
    formatted = ""
    for site, content in src_dict.items():
        sentences = re.split(r'(?<=[.!?]) +', content)
        snippet = " ".join(sentences[:5])
        formatted += f"- **{site}**:\n  {snippet}\n\n"
    return formatted

# -----------------------
# Branch definitions
# -----------------------
def enrich_with_question_and_history(prev, original):
    return {
        "sources": prev,  # raw search results
        "question": original["input"],
        "history": original.get("history", []),
        "original": original,
    }

def summarise_with_sources(data):
    summary = summarise_runnable.invoke({
        "sources": data["sources"],  # carry forward correctly
        "question": data["question"]
    })
    return {
        "summary": summary,
        "sources": data["sources"],
        "original": data["original"],
    }

def enrich_final_summary(data):
    original = data["original"]
    return {
        "input": f"""**Question:** {original['input']}  

**Verified medical information (summarised from sources):**  
{data['summary']}  

---

**Sources referenced:**  
{format_sources(data['sources'])}""",
        "history": original.get("history", []),
    }

search_branch = (
    RunnableLambda(lambda x: {"original": x, "results": medical_search(x["input"])})
    | RunnableLambda(lambda d: enrich_with_question_and_history(d["results"], d["original"]))
    | RunnableLambda(lambda d: summarise_with_sources(d))
    | RunnableLambda(lambda d: enrich_final_summary(d))
)

no_search_branch = RunnableLambda(lambda x: {"input": x["input"], "history": x.get("history", [])})

router_chain = RunnableBranch(
    (lambda x: route(x["input"]) == "search", search_branch),
    no_search_branch
)

# -----------------------
# Main LLM for final response
# -----------------------
medical_prompt = ChatPromptTemplate.from_template("""
You are a compassionate medical assistant. Answer the following question based on context if provided.
Always include disclaimers to consult a healthcare professional if needed.

Conversation history: {history}

Question and context:
{input}
""")

# Main LLM chain replacement
medical_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)

# -----------------------
# Response helper
# -----------------------
def get_medical_answer(query: str) -> str:
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."

    context = {"input": query, "history": st.session_state.memory.chat_memory.messages}
    routed_input = router_chain.invoke(context)
    final_response = medical_runnable.invoke(routed_input)
    return final_response.strip()

# -----------------------
# Streamlit UI
# -----------------------
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("Ask your medical question")
    submit = st.form_submit_button("Submit")

if submit and user_query:
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <img src="https://github.com/Vcreativelab/custom_medical_agent/blob/main/doc.gif?raw=true" 
                         width="220" style="border-radius: 10px; margin-bottom: 0.5rem;">
                    <p style="color: gray; font-size: 0.9rem;">🧠 Processing your question...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        answer = get_medical_answer(user_query)
        time.sleep(0.5)
        st.empty()

    st.markdown("### 💬 Suggestion")
    st.markdown(answer.replace("\n", "  \n"), unsafe_allow_html=True)

    # Add to memory
    st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_query))
    st.session_state.memory.chat_memory.add_message(AIMessage(content=answer))

# Collapsible chat history
if st.session_state.memory.chat_memory.messages:
    with st.expander("🩺 View Chat History", expanded=False):
        history_md = ""
        for m in st.session_state.memory.chat_memory.messages[-10:]:
            if isinstance(m, HumanMessage):
                history_md += f"**You:** {m.content}  \n"
            else:
                history_md += f"**DocBot:**  \n{m.content.replace(chr(10), '  \n')}  \n\n"
        st.markdown(history_md, unsafe_allow_html=True)
