import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embeddings import DummyEmbeddings
from src.config import CHROMA_PATH, REVIEWS_CSV_PATH


def build_vector_store():
    df = pd.read_csv(REVIEWS_CSV_PATH)

    documents = [
        Document(page_content=row["patient_experience"] + " " +
                 row["staff_behavior"] + " " +
                 row["facility_feedback"])
        for _, row in df.iterrows()
    ]

    embeddings = DummyEmbeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectordb
