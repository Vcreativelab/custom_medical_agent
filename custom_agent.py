import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any
import unicodedata

import streamlit as st
import diskcache as dc

from langchain.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableMap, RunnableSequence
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

cache = dc.Cache(os.path.join(os.getcwd(), "medical_cache"))

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
You are a medical summarisation assistant.

Summarise the following **verified medical information** clearly and concisely.

**Instructions:**
- Use **short Markdown bullet points** for key facts.
- For each key fact, include the **source in parentheses**.
- Focus on **definitions, causes, symptoms, and treatments/medications**.
- Combine overlapping information across sources (no repetition).
- Avoid quoting full paragraphs from sources — summarise meaningfully.
- Finish with a one-line **disclaimer** reminding users to consult a doctor.

---

**Collected information from sources:**
{sources}

**User question:**
{question}

Format your entire answer in Markdown.
""")

# Summarisation chain replacement
summarise_runnable = (
    summarise_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)

# -----------------------
# Language detection & translation
# -----------------------
translation_cache = dc.Cache(os.path.join(os.getcwd(), "translation_cache"))

def detect_and_translate(query: str) -> Dict[str, str]:
    """Detect language and translate non-English text to English with debug info."""
    query_key = query.strip().lower()
    if query_key in translation_cache:
        return translation_cache[query_key]

    translator_prompt = ChatPromptTemplate.from_template("""
    You are a translation assistant.
    Detect the language of this text and, if it's not English, translate it into English.
    Return strictly JSON with keys "language" and "translation".
    Text: {text}
    """)

    translator_chain = (
        translator_prompt
        | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
        | StrOutputParser()
    )

    lang, translation = "unknown", query  # fallback

    try:
        result = translator_chain.invoke({"text": query}).strip()
        # st.write("DEBUG: Raw translation output:", result)

        # Try JSON extraction first
        match = re.search(r"\{.*?\}", result, re.DOTALL)
        if match:
            raw_json = match.group(0)
            clean_json = raw_json.replace("'", '"').replace("\n", " ").strip()
            try:
                parsed = json.loads(clean_json)
            except json.JSONDecodeError:
                st.warning(f"DEBUG: JSON decode failed: {clean_json}")
                parsed = {}
                lang_match = re.search(r'"language"\s*:\s*"([^"]+)"', clean_json)
                trans_match = re.search(r'"translation"\s*:\s*"([^"]+)"', clean_json)
                if lang_match:
                    parsed["language"] = lang_match.group(1)
                if trans_match:
                    parsed["translation"] = trans_match.group(1)

            lang = parsed.get("language", "unknown").strip()
            translation = parsed.get("translation", query).strip()

    except Exception as e:
        st.warning(f"DEBUG: Translation step failed: {e}")

    # st.write("DEBUG: Parsed language:", lang)
    # st.write("DEBUG: Parsed translation:", translation)

    translation_cache[query_key] = {"language": lang, "translation": translation}
    translation_cache.expire(query_key, 60 * 60 * 24 * 7)
    return {"language": lang, "translation": translation}

# -----------------------
# Routing logic
# -----------------------
def should_search(input_text: str) -> bool:
    pattern = r"\b(treat(ment)?|symptom(s)?|drug(s)?|medication(s)?|cause(s)?|prevent(ion)?|diagnos(e|is)|test(s)?|therapy|dose|prescrib(e|ed|ing))\b"
    return bool(re.search(pattern, input_text.lower()))

def route(input_text: str):
    return "search" if should_search(input_text) else "no_search"

def format_sources(src_dict: dict) -> str:
    formatted = ""
    for site in src_dict.keys():
        domain = site.replace("www.", "")
        formatted += f"- [{domain}](https://{domain})\n"
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

📚 **Sources referenced:**  
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
You are **DocBot**, a knowledgeable and empathetic medical assistant.

Your tone should be:
- **Professional and factual**, not casual.
- **Clear and simple**, avoiding unnecessary jargon.
- **Neutral** and **evidence-based** — do not speculate or invent facts.

When answering:
- Give **concise explanations** of the condition or topic.
- If relevant, include **possible causes, symptoms, treatments, or precautions**.
- Do **not** recommend specific medications unless they are **widely accepted** first-line options.
- If unsure, clearly say that more context or clinical evaluation is needed.
- Always end with a **disclaimer** reminding users to consult a licensed healthcare professional.

---

Conversation history:
{history}

User question and context:
{input}
""")

# Main LLM chain replacement
medical_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)

# -----------------------
# Back-translation cache
# -----------------------
back_translation_cache = dc.Cache(os.path.join(os.getcwd(), "back_translation_cache"))
# -----------------------
# Get_medical_answer() back-translation block
# -----------------------
def get_medical_answer(query: str) -> str:
    """Generate a medical answer, translating back to original language if needed."""
    final_response = None # initialize to catch issues
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."
    
    # st.write("DEBUG: Query received:", query)
    
    try:
        # Detect language and translate to English if needed
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"]
        translated_query = lang_info["translation"]
        # st.write("DEBUG: Language detected:", user_lang)
        # st.write("DEBUG: Translated query:", translated_query)

        context = {"input": translated_query, "history": st.session_state.memory.chat_memory.messages}
        routed_input = router_chain.invoke(context)
        # st.write("DEBUG: Routed input:", routed_input)

        # Get or summarise response
        if isinstance(routed_input, dict) and (
            "Verified medical information" in routed_input.get("input", "") or
            "Sources referenced" in routed_input.get("input", "")
        ):
            final_response = routed_input.get("input")
        else:
            english_response = medical_runnable.invoke(routed_input)
            final_response = f"""**Question:** {query}    

**Answer:**  
{english_response}  

---

⚠️ *This information is for educational purposes only and should not replace professional medical advice.*"""

        # Translate back to user language if needed and not English
        if user_lang.lower() != "english":
            cache_key = f"{user_lang.lower()}::{final_response.strip()}"
            if cache_key in back_translation_cache:
                final_response_translated = back_translation_cache[cache_key]
            else:
                translator_back_prompt = ChatPromptTemplate.from_template("""
                You are a translation assistant.
                Translate the following English text into the language specified below.
                Preserve meaning, tone, and Markdown formatting.

                Target language: {target_lang}
                Text to translate:
                {text}
                """)
                translator_back_chain = (
                    translator_back_prompt
                    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
                    | StrOutputParser()
                )
                try:
                    final_response_translated = translator_back_chain.invoke({
                        "target_lang": user_lang,
                        "text": final_response
                    }).strip()
                    back_translation_cache[cache_key] = final_response_translated
                    back_translation_cache.expire(cache_key, 60 * 60 * 24 * 7)  # 1 week
                except Exception as e:
                    st.warning(f"⚠️ Back-translation failed, showing English answer: {e}")
                    final_response_translated = final_response

            # Optionally indicate translation
            final_response = f"*Translated from English to {user_lang}*\n\n{final_response_translated}"

    except Exception as e:
        st.error(f"⚠️ Error generating answer: {e}")
        final_response = f"⚠️ Failed to generate an answer: {e}"

    if not final_response:
        final_response = "⚠️ No answer generated."

    return final_response.strip()
    
# -----------------------
# Streamlit UI
# -----------------------
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("Ask your medical question")
    st.caption("🌍 Multilingual mode active — non-English questions are translated to English for evidence-based sources.")
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
