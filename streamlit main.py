import streamlit as st
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
import re
import os

os.environ["OPENAI_API_KEY"] = "{Your_API_Key}"

template = """
# Construction Consultancy Bot :building_construction:

Answer the following questions as best you can, but speaking as a passionate construction expert. You have access to the following tools:
{tools}
Use the following format:
Question: The question you have to answer
Thought: Your thought process in approaching the question
Action: Choose one of the available tools in [{tool_names}] for your action
Action Input: Provide the input required for the chosen tool
Observation: Describe the result obtained from the action
...(Repeat several times of the Thought/Action/Action Input/Observation as needed)
Thought: Now I have the final answer!
Final Answer: Provide your final answer from the perspective of an experienced construction professional

Let's get started!

Question: {input}
{agent_scratchpad}
"""

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
        return AgentAction(tool=action, tool_input=action_input.strip())

# Set up the Streamlit app
st.set_page_config(page_title="Construction Consultancy Bot", page_icon="🏗️")
st.title("Construction Consultancy Bot")

# Create a list of tools
tools = [
    DuckDuckGoSearchRun(
        query_prompt="What is the process of {input}?",
        result_prompt="According to my research, {output}.",
    ),
    DuckDuckGoSearchRun(
        query_prompt="What are the main challenges in {input}?",
        result_prompt="Based on my findings, the main challenges in {input} are {output}.",
    ),
    DuckDuckGoSearchRun(
        query_prompt="What are the best practices for {input}?",
        result_prompt="After studying the subject, I found that the best practices for {input} include {output}.",
    ),
]

# Create an instance of the LLMChain
chain = LLMChain()

# Create an instance of the ChatOpenAI model
model = ChatOpenAI()

# Create an instance of the AgentExecutor
executor = AgentExecutor(chain=chain, model=model)

# Set the custom prompt template and output parser
executor.prompt_template = CustomPromptTemplate(template=template, tools=tools)
executor.output_parser = CustomOutputParser()

# Main interaction loop
while True:
    # Receive user input
    user_input = st.text_input("User:", "")

    # Break if user input is empty
    if not user_input:
        break

    # Generate the response
    response = executor.execute(user_input)

    # Send the response to the user
    st.text_area("Construction Consultancy Bot:", response.return_values["output"])
