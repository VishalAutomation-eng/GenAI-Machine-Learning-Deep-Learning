from crews.market_crew import run_market_research
from crews.summary_crew import run_summary_crew

def main():
    topic = "CrewAI multi-agent frameworks"

    # 1. Web research using Serper
    research_result = run_market_research(topic)

    # 2. Summarize the web research
    summary_result = run_summary_crew(research_result.raw)

    with open("outputs/summary_output.md", "w", encoding="utf-8") as f:
        f.write(summary_result.raw)

    print("✅ Web research + summary completed")

if __name__ == "__main__":
    main()
