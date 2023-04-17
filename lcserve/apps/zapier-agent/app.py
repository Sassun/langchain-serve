from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit



from lcserve import serving


@serving
def ask(question: str) -> str:
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent.run(question)