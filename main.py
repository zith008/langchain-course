from langchain_core.tools import render_text_description
from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, ReActSingleInputOutputParser
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse



@tool
def get_text_length(text:str) -> int:
    """Get the length of the text by characters"""
    text = text.strip("'\n").strip('"')

    return len(text)




if __name__ == "__main__":
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

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

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template = template).partial(
    tools = render_text_description(tools), tool_names= ",".join([t.name for t in tools])
    )


    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])

    agent = {"input": lambda x:x["input"]} | prompt | llm | ReActSingleInputOutputParser()


    res = agent.invoke({"input": "What is the length in characters of the text 'DOG' in characters?"})
    print(res)




