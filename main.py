from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun 

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "Your_api_key"

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
Final Answer: Provide your final answer from the perspective of an experienced construction professional

Let's get started!
Question: {input}
{agent_scratchpad}"""

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
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    search = DuckDuckGoSearchRun().run(f"site:sweets.construction.com {input_text}")
    return search

def search_supplier(input_text):
    search = DuckDuckGoSearchRun().run(f"site:thomasnet.com {input_text}")
    return search

def search_material(input_text):
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
            name = "Search supplier",
            func=search_supplier,
            description="useful for when you need to search for construction suppliers"
        ),
        Tool(
            name = "Search material",
            func=search_material,
            description="useful for when you need to search for construction materials"
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
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]


    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=True)
    return agent_executor


# Creating an instance of the agent_executor
agent_executor = agent()
