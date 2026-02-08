import requests
from src.config import OLLAMA_URL, MODEL

def generate_sql(prompt, user_query):
    payload = {
        "model": MODEL,
        "prompt": f"{prompt}\nUser Query: {user_query}",
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"].replace("```sql", "").replace("```", "").strip()
