import json
import ast
from .vision_client import call_vision_model
from .image_utils import encode_image

def extract_json(text: str):
    if not isinstance(text, str):
        raise TypeError("Model output is not text")

    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON found in output")

    block = text[start:end + 1]

    # Try strict JSON
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    # Recover from broken JSON (unquoted keys etc.)
    try:
        return ast.literal_eval(block)
    except Exception:
        raise ValueError("Failed to parse JSON")

def generate_metadata(image_path: str, prompt: str):
    image_b64 = encode_image(image_path)

    strict_prompt = f"""
{prompt}

STRICT RULES:
- Output ONLY valid JSON
- Keys must be double-quoted
- No explanation
- No markdown
"""

    raw = call_vision_model(strict_prompt, image_b64)
    return extract_json(raw)
