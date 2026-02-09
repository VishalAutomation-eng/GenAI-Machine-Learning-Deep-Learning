from crewai import Task

def create_plan_task(agent, topic):
    return Task(
        description=f"Create a detailed content outline for topic: {topic}",
        expected_output="Structured outline with sections and key points.",
        agent=agent
    )

def create_write_task(agent):
    return Task(
        description="Write content based on the provided outline.",
        expected_output="Well-structured markdown article.",
        agent=agent
    )

def create_edit_task(agent):
    return Task(
        description="Edit the content for clarity, grammar, and consistency.",
        expected_output="Final publication-ready markdown content.",
        agent=agent
    )
