import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

def build_vector_db(json_dir, db_dir):
    embedding_fn = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    documents = []

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)
            desc = data["description"]

            text = f"""
Subject: {desc.get('main_subject')}
Description: {desc.get('detailed_description')}
Keywords: {', '.join(desc.get('search_keywords', []))}
Visual Elements: {', '.join(desc.get('visual_elements', []))}
"""

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "file_path": data.get("file_path"),
                        "file_name": file
                    }
                )
            )

    if not documents:
        print("⚠️ No documents found. Skipping embeddings.")
        return

    Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=db_dir
    )

    print(f"✅ Vector DB created with {len(documents)} documents")
