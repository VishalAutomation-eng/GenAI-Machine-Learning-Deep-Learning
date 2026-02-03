import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODELS_DIR = "/home/vkpande/GenAI-Machine-Learning-Deep-Learning/Day10-face_generator_inference/models/models/models/generator_100.h5"
OUTPUT_GIF = "/home/vkpande/GenAI-Machine-Learning-Deep-Learning/Day10-face_generator_inference/outputd/evolution.gif"

noise = tf.random.normal([1, 100])
frames = []

# Loop through saved generator checkpoints
for i in range(1, 8):
    model_path = f"{MODELS_DIR}/generator_{i}00.h5"
    print(f"Loading {model_path}")

    generator = tf.keras.models.load_model(model_path, compile=False)

    with tf.device("/CPU:0"):
        generated_image = generator(noise, training=False)

    generated_image = (generated_image + 1) / 2.0
    image = generated_image[0].numpy()

    # Save frame temporarily
    frame_path = f"frame_{i}.png"
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(frame_path)
    plt.close()

    frames.append(imageio.imread(frame_path))
    os.remove(frame_path)

# Save GIF
imageio.mimsave(OUTPUT_GIF, frames, fps=1)

print(f"🎉 Evolution GIF saved to {OUTPUT_GIF}")
