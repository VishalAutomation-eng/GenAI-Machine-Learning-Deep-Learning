from crewai import Crew, Process, LLM
from agents.summary_agent import create_summary_agent
from tasks.summary_task import create_summary_task
import os

def run_summary_crew(content: str):
    llm = LLM(
        model=os.getenv("OLLAMA_MODEL"),          
        api_base=os.getenv("OLLAMA_BASE_URL"),    
        api_key="ollama",                         # dummy
    )

    agent = create_summary_agent(llm)
    task = create_summary_task(agent, content)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
