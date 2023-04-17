from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, AgentExecutor, Tool, ZeroShotAgent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit


from lcserve import serving


@serving
def ask(question: str) -> str:
    llm = OpenAI(temperature=0)

    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or search information from google. You should ask targeted questions"
        )
    ]
   # for i in toolkit.get_tools():
   #     tools.append(i)
   # return tools 
   # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent.run(question)