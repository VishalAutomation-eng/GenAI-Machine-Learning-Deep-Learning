import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --------------------------------------------------
# Force CPU (safe on macOS)
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# Resolve paths safely (OS-independent)
# --------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent   # Day10-face_generator_inference/

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputd"

WEIGHTS_PATH = MODELS_DIR / "generator_10.h5"
OUTPUT_PATH = OUTPUTS_DIR / "single_image.png"

print("🔍 Loading FULL model from:", WEIGHTS_PATH)

# --------------------------------------------------
# Validation
# --------------------------------------------------
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"❌ Model file not found.\n"
        f"Expected at: {WEIGHTS_PATH}\n"
        f"Please place generator_10.h5 inside the models/ directory."
    )

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
plt.figure(figsize=(4, 4))
plt.imshow(generated_image[0])
plt.axis("off")
plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0)
plt.show()

print(f"✅ Image saved at: {OUTPUT_PATH}")
