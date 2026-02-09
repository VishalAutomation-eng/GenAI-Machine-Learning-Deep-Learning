from crewai import Agent

def create_planner_agent(llm):
    return Agent(
        role="Content Planner",
        goal="Plan structured, SEO-optimized content",
        backstory="You specialize in planning high-quality educational content.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

def create_writer_agent(llm):
    return Agent(
        role="Content Writer",
        goal="Write clear and engaging content",
        backstory="You convert outlines into readable and engaging articles.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

def create_editor_agent(llm):
    return Agent(
        role="Content Editor",
        goal="Edit and polish content for clarity and accuracy",
        backstory="You ensure grammatical correctness and professional tone.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
