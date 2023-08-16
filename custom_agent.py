# Imports
import streamlit as st
import langchain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import os
import re


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Page title
st.set_page_config(page_title="What's up Doc? 🤖🩺")
st.title("What's up Doc? 🤖🩺")


# import API keys
if __name__ == '__main__':
    import os
    #from dotenv import load_dotenv, find_dotenv
    #load_dotenv(find_dotenv(), override=True)


    with st.sidebar:
        k_value = st.number_input('K value', min_value=1, max_value=10, value=3, on_change=clear_history)
        api_key = st.text_input('OpenAI API Key', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        if not api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')

    # Define which tools the agent can use to answer user queries
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

    # Set up the base template
    template_with_history = """Answer the following questions as best you can, 
    but speaking as compassionate medical professional. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin! Remember to speak as a compassionate medical professional when giving your final answer. 
    If the condition is serious advise they speak to a doctor.
    
    Previous conversation history:
    {history}
    
    New question: {input}
    {agent_scratchpad}"""


    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)


    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
    )


    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


    output_parser = CustomOutputParser()


    # Generate LLM response
    def generate_response(input_query, k_value=3):
        # Instantiating llm model form OpenAI
        llm = OpenAI(temperature=0)

        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
        # looping over different tools
        tool_names = [tool.name for tool in tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        # Memory keeps a list of the interactions of the conversation over time.
        # It only uses the last K interactions
        from langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=k_value)

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                            tools=tools,
                                                            verbose=True,
                                                            memory=memory
                                                            )
        # Perform Query using the Agent
        response = agent_executor.run(input_query)
        return response


    with st.form(key="my_query", clear_on_submit=True):
        q = st.text_input("Ask your question", key="user_question")
        submit_button = st.form_submit_button("Submit")


    import time
    if submit_button:

        gif = st.image('doc.gif', caption='Your question is being processed. Please wait a few seconds')
        answer = generate_response(q, k_value)
        gif.empty()
        time.sleep(0.5)
        st.text_area('**Suggestion**', value=answer, height=200)
        st.divider()

        # if there's no chat history in the session state, create it
        if 'history' not in st.session_state:
            st.session_state.history = ''
        # # Concatenating question with answer
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-" * 130} \n {st.session_state.history}'
        h = st.session_state.history
        time.sleep(1)
        st.text_area(label='Chat history', value=h, key='history', height=400)


