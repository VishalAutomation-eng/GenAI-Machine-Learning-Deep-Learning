from pathlib import Path
import tensorflow as tf
import imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_GIF = PROJECT_ROOT / "outputd" / "evolution.gif"

OUTPUT_GIF.parent.mkdir(exist_ok=True)

# Get all available generator models (sorted)
model_paths = sorted(MODELS_DIR.glob("generator_*.h5"))

if not model_paths:
    raise FileNotFoundError("❌ No generator_*.h5 models found in models/ directory")

print("📦 Found models:")
for p in model_paths:
    print("  -", p.name)

noise = tf.random.normal([1, 100])
frames = []

for model_path in model_paths:
    print("🔄 Loading", model_path.name)

    generator = tf.keras.models.load_model(model_path, compile=False)
    image = generator(noise, training=False)
    image = (image + 1) / 2.0
    image = tf.clip_by_value(image, 0.0, 1.0)

    frame = (image[0].numpy() * 255).astype("uint8")
    frames.append(frame)

imageio.mimsave(OUTPUT_GIF, frames, fps=1)
print(f"🎉 Evolution GIF saved at: {OUTPUT_GIF}")
