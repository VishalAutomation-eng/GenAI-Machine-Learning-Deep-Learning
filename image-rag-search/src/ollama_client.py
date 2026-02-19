import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload, timeout=300)
    res.raise_for_status()

    data = res.json()

    response = data.get("response")

    if not response:
        raise ValueError("Empty response from Ollama")

    if not isinstance(response, str):
        raise TypeError(f"Ollama response is not string: {type(response)}")

    return response
