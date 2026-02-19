from crewai import Crew, Process

def run_support_crew(agents, tasks):
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    ).kickoff()
