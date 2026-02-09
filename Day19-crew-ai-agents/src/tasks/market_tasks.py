from crewai import Task

def create_web_research_task(agent, topic: str):
    return Task(
        description=(
            f"Search the web for the latest information on: {topic}\n\n"
            "Rules:\n"
            "- Use web search results\n"
            "- Focus on factual, recent data\n"
            "- Summarize key findings\n"
            "- Avoid speculation"
        ),
        expected_output=(
            "A concise research summary with key findings, trends, and sources."
        ),
        agent=agent
    )
