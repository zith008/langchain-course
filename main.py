
from dotenv import load_dotenv

load_dotenv()

from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch


from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse



@tool
def get_text_length(text:str) -> int:
    """Get the length of the text by characters"""
    text = text.strip("'\n").strip('"')

    return len(text)




if __name__ == "__main__":
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    tools = [TavilySearch()]
    llm = ChatOpenAI(model = "gpt-3.5-turbo")
    # llm = ChatAnthropic(model = "claude-3-5-sonnet-20240620")

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools)

    res = agent_executor.invoke(
        {
            "input": "what is the weather in dubai right now?, compare it with singapore , output should be in celcius"
        }
    )

    print(res)

