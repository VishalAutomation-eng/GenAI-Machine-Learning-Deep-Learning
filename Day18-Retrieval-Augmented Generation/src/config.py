import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")

CHROMA_PATH = "chroma_data"
REVIEWS_CSV_PATH = "data/reviews.csv"
