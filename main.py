import streamlit as st
from langchain.langchain import LangChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents.agent import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from pydantic import Field
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
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    input_variables: List[str]
    
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
class ChatOpenAI(LangChain):
    prompt_template_class = CustomPromptTemplate
    output_parser_class = CustomOutputParser
    
    def __init__(self, agent_executor, **kwargs):
        super().__init__(**kwargs)
        self.agent_executor = agent_executor
    
    def __call__(self, query: str, **kwargs):
        return self.agent_executor(query, **kwargs)

def get_agent_executor():
    return AgentExecutor(
        ChatOpenAI(
            os.environ.get("OPENAI_API_KEY", ""),
            gpt3_model="gpt-3.5-turbo",
            gpt3_model_owner="openai",
            templates=[template],
            tools=[
                Tool(
                    search_online,
                    input_variables=["input_text"],
                    description="Search online for information",
                ),
                Tool(
                    search_equipment,
                    input_variables=["input_text"],
                    description="Search equipment listings",
                ),
                Tool(
                    search_general,
                    input_variables=["input_text"],
                    description="Search in general",
                ),
            ],
        )
    )

async def main(query: str):
    agent_executor = get_agent_executor()
    agent = LLMSingleActionAgent(agent_executor)
    async with agent as runner:
        async for response in runner.iter_inferences([query]):
            yield response

st.title("Construction Expert Chat")

query = st.text_input("Ask a question:")
if query:
    result = asyncio.run(main(query))
    response = result[0]
    if isinstance(response, AgentAction):
        st.write(response.tool)
    elif isinstance(response, AgentFinish):
        st.write(response.return_values["output"])
