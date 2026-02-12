import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = react_prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors=True 
)

chain = agent_executor

def main():
    print("hello from langchain")
    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ai engineer using langchain in singapore on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
