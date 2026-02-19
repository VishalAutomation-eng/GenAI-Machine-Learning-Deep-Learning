import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
import os
from models.generator_model import build_generator
from tensorflow.keras import layers


LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10
SAVE_INTERVAL = 10

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load CIFAR-10 dataset
(x_train, _), _ = tf.keras.datasets.cifar10.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = tf.image.resize(x_train, (64, 64))
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(BATCH_SIZE)

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding="same", input_shape=(64, 64, 3)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator(LATENT_DIM)
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(fake_images, training=True)

        gen_loss = loss_fn(tf.ones_like(fake_out), fake_out)
        disc_loss = (
            loss_fn(tf.ones_like(real_out), real_out) +
            loss_fn(tf.zeros_like(fake_out), fake_out)
        )

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

for epoch in range(1, EPOCHS + 1):
    for batch in dataset:
        train_step(batch)

    if epoch % SAVE_INTERVAL == 0:
        path = os.path.join(MODELS_DIR, f"generator_{epoch}.h5")
        generator.save(path)
        print(f"✅ Saved {path}")
