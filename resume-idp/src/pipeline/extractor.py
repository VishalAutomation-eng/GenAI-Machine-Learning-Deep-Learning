import json
import os
from tqdm import tqdm

from src.prompt.resume_prompt import build_resume_prompt
from src.llm.ollama_client import call_ollama
from src.config.settings import OUTPUT_PATH

def extract_from_resumes(df):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        resume_text = row["Resume_str"]

        prompt = build_resume_prompt(resume_text)
        response = call_ollama(prompt)

        try:
            data = json.loads(response)
        except Exception:
            print(f"❌ JSON error at row {idx}")
            continue

        output_file = os.path.join(OUTPUT_PATH, f"resume_{idx}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
            


