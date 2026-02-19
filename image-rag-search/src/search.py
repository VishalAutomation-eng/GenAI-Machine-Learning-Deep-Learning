from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def search(query, db_dir, top_k=3):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )

    return db.similarity_search(query, k=top_k)
