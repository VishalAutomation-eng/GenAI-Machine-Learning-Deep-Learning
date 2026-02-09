from crewai import Agent

def create_summary_agent(llm):
    return Agent(
        role="AI Output Summarizer",
        goal="Summarize outputs into exactly 30–40 lines",
        backstory=(
            "You are a technical summarization expert. "
            "You produce concise, structured summaries without losing meaning."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
