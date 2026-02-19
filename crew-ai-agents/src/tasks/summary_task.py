from crewai import Task

def create_summary_task(agent, content):
    return Task(
        description=(
            "Summarize the following content into STRICTLY 30–40 lines.\n"
            "Rules:\n"
            "- Mention agents, tasks, tools, workflow\n"
            "- Technical tone only\n"
            "- No filler text\n\n"
            f"{content}"
        ),
        expected_output="A concise 30–40 line technical summary.",
        agent=agent
    )
