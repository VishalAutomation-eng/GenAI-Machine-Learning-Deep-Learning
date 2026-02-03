from tensorflow import keras

def load_cifar100():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()

    # Normalize (same as Fashion-MNIST)
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    return train_images, train_labels, test_images, test_labels
