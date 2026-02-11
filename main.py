import os

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch


from langchain_ollama import ChatOllama

@tool
def search(query: str) -> str:
    """Tool that searches the web for the query.
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model = "gpt-3.5-turbo")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools = tools)
def main():
    
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the singapore on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()
