import os
import json
from tqdm import tqdm
from .metadata_generator import generate_metadata

def process_images(image_dir, output_dir, prompt):
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(image_dir)

    for img in tqdm(images):
        if not img.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        json_path = os.path.join(
            output_dir, img.rsplit(".", 1)[0] + ".json"
        )

        if os.path.exists(json_path):
            continue

        try:
            data = generate_metadata(
                os.path.join(image_dir, img),
                prompt
            )
        except Exception as e:
            print(f"❌ Failed for {img}: {e}")
            continue

        with open(json_path, "w") as f:
            json.dump(
                {
                    "file_path": os.path.join(image_dir, img),
                    "description": data
                },
                f,
                indent=4
            )
