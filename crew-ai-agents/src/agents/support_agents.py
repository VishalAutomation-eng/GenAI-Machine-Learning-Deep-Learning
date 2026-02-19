from crewai import Agent

def create_support_agent(llm):
    return Agent(
        role="Senior Support Representative",
        goal="Provide accurate and friendly customer support",
        backstory="You handle critical customer issues with clarity and empathy.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

def create_qa_agent(llm):
    return Agent(
        role="Support QA Specialist",
        goal="Ensure response quality and correctness",
        backstory="You review support answers for accuracy and tone.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
