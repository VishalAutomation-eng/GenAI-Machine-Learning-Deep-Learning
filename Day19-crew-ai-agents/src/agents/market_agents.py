from crewai import Agent
from crewai_tools import SerperDevTool
import os

def create_market_researcher(llm):
    search_tool = SerperDevTool(
        api_key=os.getenv("SERPER_API_KEY")
    )

    return Agent(
        role="Market Researcher",
        goal="Search the web and collect accurate, up-to-date information",
        backstory=(
            "You are an expert market and technology researcher. "
            "You use web search to find reliable, recent information "
            "and summarize it in a structured technical format."
        ),
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
