import streamlit as st
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents.agent import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
import re
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import os
import chainlit as cl

api_key = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = api_key
template = """
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
Final Answer: {Final AnswerTool}
"""

class CustomPromptTemplate(StringPromptTemplate):
    
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

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.?)\nAction\s\d*\s*Input\s*\d*\s*:[\s](.)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def search_online(input_text):
    search = DuckDuckGoSearchRun().run(f"site:Americanbuildsupply.com {input_text}")
    return search

def search_equipment(input_text):
    search = DuckDuckGoSearchRun().run(f"site:machinerytrader.com {input_text}")
    return search

def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search

@cl.langchain_factory(use_async=False)
def agent():
    tools = [
        Tool(
            name = "Search general",
            func=search_general,
            description="useful for when you need to answer general construction-related questions"
        ),
        Tool(
            name = "Search construction",
            func=search_online,
            description="useful for when you need to search for construction-related information online"
        ),
        Tool(
            name = "Search equipment",
            func=search_equipment,
            description="useful for when you need to search for construction equipment"
        )
    ]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor

# Create an instance of the agent_executor
agent_executor = agent()

# Streamlit app
def main():
    st.title("Construction Consultancy Bot👷‍♂️👷‍♀️🚧")
    st.write("Ask construction-related questions and get expert advice!")

    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if question:
            response = agent_executor()
            st.write(response["output"])
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()

