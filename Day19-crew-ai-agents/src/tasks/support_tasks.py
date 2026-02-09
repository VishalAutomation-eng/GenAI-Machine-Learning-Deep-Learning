from crewai import Task

def create_support_task(agent, inquiry):
    return Task(
        description=f"Resolve customer inquiry:\n{inquiry}",
        expected_output="Complete and friendly customer response.",
        agent=agent
    )

def create_qa_task(agent):
    return Task(
        description="Review support response for quality and accuracy.",
        expected_output="Final customer-ready support response.",
        agent=agent
    )
