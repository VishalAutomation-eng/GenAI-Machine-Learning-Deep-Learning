import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ---- LLM CONFIG ----
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")

if not OLLAMA_URL or not MODEL:
    raise ValueError("❌ OLLAMA_URL or MODEL not set in .env")

# ---- DATABASE CONFIG ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "employees.db")
