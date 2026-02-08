import streamlit as st
from src.vector_store import build_vector_store
from src.rag_chain import run_rag

st.set_page_config(page_title="Hospital Review RAG Bot", layout="wide")

st.title("🏥 Hospital Review Helper (RAG + Ollama)")

@st.cache_resource
def load_db():
    vectordb = build_vector_store()
    return vectordb.as_retriever(k=10)

retriever = load_db()

question = st.text_input("Ask a question about hospital reviews:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = run_rag(question, retriever)
            st.markdown("### 🤖 Answer")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
