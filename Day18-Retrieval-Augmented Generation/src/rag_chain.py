from src.llm import ollama_llm
from src.prompt import REVIEW_PROMPT


def run_rag(question: str, retriever) -> str:
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = REVIEW_PROMPT.format(
        context=context,
        question=question
    )

    return ollama_llm(prompt)
