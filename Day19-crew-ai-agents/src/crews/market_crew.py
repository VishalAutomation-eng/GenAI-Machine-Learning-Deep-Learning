from crewai import Crew, Process, LLM
from agents.market_agents import create_market_researcher
from tasks.market_tasks import create_web_research_task
import os

def run_market_research(topic: str):
    llm = LLM(
        model=os.getenv("OLLAMA_MODEL"),
        api_base=os.getenv("OLLAMA_BASE_URL"),
        api_key="ollama"
    )

    researcher = create_market_researcher(llm)
    task = create_web_research_task(researcher, topic)

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
