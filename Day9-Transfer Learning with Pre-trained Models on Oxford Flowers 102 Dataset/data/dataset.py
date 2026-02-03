import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_flowers102():
    dataset, info = tfds.load(
        'oxford_flowers102:2.1.1',
        with_info=True,
        as_supervised=True
    )

    train_ds = dataset['train']
    val_ds = dataset['validation']
    test_ds = dataset['test']

    def preprocess(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    train_ds = (
        train_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, info
