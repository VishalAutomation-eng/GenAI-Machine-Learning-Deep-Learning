import os
from src.ingest_metadata import process_images
from src.build_embeddings import build_vector_db

PROMPT_PATH = "prompts/image_metadata_prompt.txt"

with open(PROMPT_PATH) as f:
    PROMPT = f.read()

process_images(
    image_dir="data/images",
    output_dir="data/metadata",
    prompt=PROMPT
)

if os.listdir("data/metadata"):
    build_vector_db(
        json_dir="data/metadata",
        db_dir="chromadb"
    )
else:
    print("⚠️ No metadata generated. Skipping embeddings.")
