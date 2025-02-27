import streamlit as st
import langchain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import os
import re
import time

# Gemini imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory

# Rate limiting variables (Gemini-1.5-pro: 2 requests per minute, placeholder for token limits)
RATE_LIMIT_SECONDS = 60  # 1 minute window
MAX_REQUESTS_PER_MINUTE = 2
MAX_TOKENS_PER_MINUTE = 32000  # Placeholder; token usage not actively tracked here.
request_times = []
token_count = 0  # For demonstration purposes

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Page title
st.set_page_config(page_title="What's up Doc? 🤖🩺")
st.title("What's up Doc? 🤖🩺")

# API Key Setup
if __name__ == '__main__':
    with st.sidebar:
        k_value = st.number_input('K value', min_value=1, max_value=10, value=3, on_change=clear_history)
        use_secrets = st.toggle("Use Streamlit Secrets for API Key", value=True)

        if use_secrets:
            try:
                gemini_api_key = st.secrets["GOOGLE_API_KEY"]
                os.environ['GOOGLE_API_KEY'] = gemini_api_key
                genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            except Exception as e:
                st.error(f"Error configuring Gemini API: {e}")
                gemini_api_key = st.text_input('Gemini API Key', type='password')
                if gemini_api_key:
                    os.environ['GOOGLE_API_KEY'] = gemini_api_key
                    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
                else:
                    st.stop()
        else:
            gemini_api_key = st.text_input('Gemini API Key', type='password')
            if gemini_api_key:
                os.environ['GOOGLE_API_KEY'] = gemini_api_key
                genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            else:
                st.warning('Please enter your Gemini API key!', icon='⚠')
                st.stop()

# Define the search tool
search = DuckDuckGoSearchRun()

def duck_wrapper(input_text):
    search_results = search.run(f"site:webmd.com {input_text}")
    return search_results

tools = [
    Tool(
        name="Search WebMD",
        func=duck_wrapper,
        description="useful for when you need to answer medical and pharmacological questions"
    )
]

# Sidebar for response style selection
with st.sidebar:
    response_style = st.radio("Response Style", ["Concise", "Detailed"])

# Initialize session state variables if not already set
if "history" not in st.session_state:
    st.session_state.history = ""

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3)

# Ensure conversation length control
if 'conversation_length' not in st.session_state:
    st.session_state.conversation_length = 3  # Start with window size of 3

# Create memory with a dynamic window size and expand after each query
memory = ConversationBufferWindowMemory(k=st.session_state.conversation_length)
st.session_state.conversation_length += 1
st.session_state.conversation_length = min(st.session_state.conversation_length, 10)  # Cap at 10

# Set up the base template with few-shot examples
template_with_history = """You are an AI-powered medical assistant.  
Your goal is to provide **clear, accurate, and compassionate** medical guidance based on **reliable sources**.  
You should always prioritize **evidence-based medicine** and **encourage professional consultation** when necessary.  

### **Rules to Follow:**  
1️⃣ **Context Awareness:** Assume follow-up questions refer to the last discussed condition unless stated otherwise. Clarify when needed.  
2️⃣ **Clarity & Simplicity:** Use **accessible language** and break down complex terms.  
3️⃣ **Safety First:** If symptoms suggest a serious condition, **urge the user to seek professional medical care**. Never provide medication dosages.  
4️⃣ **Honesty in Uncertainty:** If unsure, **DO NOT GUESS**—say that more research is needed.  

---

### **Example Responses for Guidance:**  

#### **Basic Definitions**  
**Question:** What is pneumonia?  
**Thought:** The user wants a definition. I should provide a concise medical explanation.  
**Final Answer:** Pneumonia is an infection that inflames the air sacs in the lungs. It can be caused by bacteria, viruses, or fungi.  

#### **Treatment Guidance**  
**Question:** How is pneumonia treated?  
**Thought:** The user is asking about treatment. I should include medication options and when to seek medical help.  
**Final Answer:** Treatment depends on the cause—bacterial pneumonia needs antibiotics, while viral pneumonia requires supportive care.  

#### **Handling Ambiguous Questions**  
**Question:** Is it dangerous?  
**Thought:** The user did not specify what "it" refers to. I should ask for clarification if needed, or assume they are referring to the last mentioned condition.  
**Final Answer:** If you are referring to pneumonia, it can be serious if untreated, especially for older adults or those with weak immune systems.  

#### **Recognizing Urgent Symptoms**  
**Question:** I have chest pain and trouble breathing. What should I do?  
**Thought:** The user is describing potentially life-threatening symptoms. I must prioritize their safety.  
**Final Answer:** Your symptoms could indicate a serious medical emergency, such as a heart attack or severe pneumonia. **Seek emergency medical attention immediately.**  

#### **Preventive Care**  
**Question:** How can I prevent the flu?  
**Thought:** The user is asking about flu prevention. I should provide evidence-based recommendations.  
**Final Answer:** To prevent the flu, get vaccinated annually, wash hands frequently, and avoid close contact with sick individuals.  

#### **Explaining Medical Mechanisms**  
**Question:** Why does diabetes cause fatigue?  
**Thought:** The user is asking about the connection between diabetes and fatigue. I should explain how insulin resistance affects energy levels.  
**Final Answer:** Diabetes can cause fatigue due to unstable blood sugar levels, insulin resistance, and inflammation, which impact the body's ability to use energy efficiently.  

---

### **Context:**  
**Previous conversation history:**  
{history}  

**New question:** {input}  
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if Final Answer is present; if so, use it.
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Improved regex to capture action and optional action input.
        regex = r"Action\s*\d*\s*:\s*(.*?)\n(?:Action\s*\d*\s*Input\s*\d*\s*:\s*(.*))?"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip() if match.group(1) else None
            action_input = match.group(2).strip() if match.group(2) else ""
            if action:
                return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        # Fallback: return the raw output.
        return AgentFinish(return_values={"output": llm_output.strip()}, log=llm_output)

output_parser = CustomOutputParser()

def is_rate_limited():
    global request_times
    now = time.time()
    # Remove timestamps older than RATE_LIMIT_SECONDS
    request_times = [t for t in request_times if now - t < RATE_LIMIT_SECONDS]
    return len(request_times) >= MAX_REQUESTS_PER_MINUTE

def get_last_condition(history):
    matches = re.findall(r"Q:\s*(.*?)\n", history)
    return matches[-1] if matches else None  # Get the most recent question

def generate_response(input_query, k_value=3):
    global request_times

    if is_rate_limited():
        st.warning(f"Rate limit exceeded. Please wait {RATE_LIMIT_SECONDS} seconds.")
        return "The system is currently overloaded. Please try again later."

    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.0)
        llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
        tool_names = [tool.name for tool in tools]
        memory = st.session_state.memory  # Use stored memory

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory
        )

        if re.match(r"^(how|what|why|when|where|can it|is it)", input_query, re.IGNORECASE):
            last_condition = get_last_condition(st.session_state.history)
            if last_condition:
                input_query = f"{input_query} (Referring to: {last_condition})"

        request_times.append(time.time())
        response = agent_executor.run(input_query)
        return response

    except Exception as e:
        st.error(f"An error occurred during the request: {e}")
        return "An error occurred while processing the request."

with st.form(key="my_query", clear_on_submit=True):
    q = st.text_input("Ask your question", key="user_question")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Display processing GIF
    gif_runner = st.image("doc.gif", caption="Processing your request", output_format="auto")
    answer = generate_response(q, k_value)
    time.sleep(0.5)
    gif_runner.empty()
    st.text_area('**Suggestion**', value=answer, height=200)
    st.session_state.history = f'Q: {q} \nA: {answer} \n{"-" * 130} \n{st.session_state.history}'
    st.text_area(label='Chat history', key='history', height=400)
