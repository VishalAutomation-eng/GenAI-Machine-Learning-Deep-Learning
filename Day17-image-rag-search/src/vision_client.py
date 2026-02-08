import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def call_vision_model(prompt: str, image_base64: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=300
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]
