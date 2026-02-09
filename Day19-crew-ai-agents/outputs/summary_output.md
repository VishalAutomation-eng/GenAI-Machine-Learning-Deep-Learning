1. CrewAI is an open‑source Python library for building crews of LLM agents that collaborate on multi‑step tasks.  
2. Repository https://github.com/crewAI/crewAI (last commit 2024‑09‑10); docs https://crewai.readthedocs.io/en/latest/getting_started.html (2024‑09‑05).  
3. Version 0.5.0 introduces dynamic role assignment, allowing agents to swap roles based on outcomes or external signals.  
4. Asynchronous DAG engine enables parallel execution of independent tasks, reducing overall runtime.  
5. Shared memory layer (MemoryStore) supports in‑memory, Redis, or PostgreSQL back‑ends for cross‑agent context.  
6. Built‑in evaluation loops provide metrics such as factual accuracy and relevance, feeding back into subsequent decisions.  
7. API‑agnostic LLMProvider plug‑in abstracts OpenAI, Anthropic, Cohere, and custom model integrations.  
8. ToolProvider interface lets agents invoke external tools (web search, DB queries, etc.) within the workflow.  
9. Declarative DSL lets users define crews in a few Python lines, specifying agents, tasks, and dependencies.  
10. Cost‑optimization utilities track token usage and enforce budget caps for large‑scale crews.  
11. Core stack: Python 3.9+, asyncio event loop, DAG scheduler, abstract MemoryStore, plug‑in architecture.  
12. Execution model supports both synchronous and asynchronous agents via the same event loop.  
13. MemoryStore interface is pluggable; default implementations include volatile, Redis, and SQL stores.  
14. Workflow engine resolves task dependencies, schedules parallel nodes, and monitors execution state.  
15. Testing pipeline uses PyTest, GitHub Actions, mypy type checking, and automated release verification.  
16. Compared to LangChain, CrewAI offers first‑class crew orchestration and built‑in shared memory.  
17. Unlike Auto‑GPT, CrewAI provides parallel DAG execution and dynamic role swapping out of the box.  
18. OpenAI function‑calling baseline lacks crew abstraction, role management, and shared memory features.  
19. Community metrics: 2.4 k GitHub stars (+35 % YoY), 150+ active forum threads/month (Sept 2024).  
20. Enterprise pilots reported in FinTech, HealthTech, Knowledge Management, Legal, and Education sectors.  
21. FinTech crew (data‑fetcher → compliance‑checker → summarizer) cut manual review time by 45 % and reached >98 % accuracy.  
22. HealthTech crew (search → medical‑knowledge → risk‑assessment) accelerated triage decisions and improved documentation consistency.  
23. Knowledge‑base update crew (crawler → extractor → fact‑checker) achieved <5 % error rate with token‑cap cost control.  
24. Education grading crew (grader → feedback generator → plagiarism detector) scaled to >10 k submissions per semester.  
25. Over 120 public GitHub repositories reference CrewAI for research assistants, code review bots, and data‑analysis pipelines.  
26. Parallel task execution is realized by marking DAG nodes as independent; the engine schedules them on the asyncio loop.  
27. Dynamic role assignment is configured via `role_switch_policy` callbacks that trigger on evaluation metric thresholds.  
28. Shared memory reads/writes use `memory_store.set(key, value)` and `memory_store.get(key)` accessible to any agent in the crew.  
29. Evaluation loops are attached to tasks via `evaluator` objects that compute scores and optionally re‑route workflow.  
30. Cost‑optimization middleware intercepts LLM API calls, aggregates token counts, and aborts execution when caps are reached.  
31. LLMProvider plug‑in requires `generate(prompt, **kwargs)` implementation; custom providers can be registered at runtime.  
32. ToolProvider plug‑in follows `execute(tool_name, args)` signature; default tools include web‑search and SQL query executor.  
33. The DSL syntax example: `Crew(agents=[...], tasks=[...], memory=RedisMemory())` defines the entire workflow declaratively.  
34. Release notes v0.5.0 (2024‑08‑20) detail the parallel DAG engine, role‑swap API, and memory store extensions.  
35. CrewAI’s roadmap emphasizes scaling crews beyond 50 agents, richer evaluators, and tighter integration with observability platforms.