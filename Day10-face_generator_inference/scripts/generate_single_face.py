import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --------------------------------------------------
# Force CPU
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# Resolve paths safely
# --------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

MODELS_DIR = Path("/home/vkpande/GenAI-Machine-Learning-Deep-Learning/Day10-face_generator_inference/models")
OUTPUTS_DIR = Path("/home/vkpande/GenAI-Machine-Learning-Deep-Learning/Day10-face_generator_inference/outputd")

WEIGHTS_PATH = MODELS_DIR / "generator_100.h5"
OUTPUT_PATH = OUTPUTS_DIR / "single_image.png"

print("🔍 Loading FULL model from:", WEIGHTS_PATH)

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"❌ Model file not found: {WEIGHTS_PATH}")

OUTPUTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load FULL model
# --------------------------------------------------
generator = tf.keras.models.load_model(WEIGHTS_PATH, compile=False)

# --------------------------------------------------
# Generate image
# --------------------------------------------------
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Rescale [-1, 1] → [0, 1]
generated_image = (generated_image + 1) / 2.0

# --------------------------------------------------
# Save output
# --------------------------------------------------
plt.imshow(generated_image[0])
plt.axis("off")
plt.savefig(OUTPUT_PATH)
plt.show()

print(f"✅ Image saved at: {OUTPUT_PATH}")
