from data_loader import load_cifar100
from cnn_model import build_cnn_model

# Load data
train_images, train_labels, test_images, test_labels = load_cifar100()

# Build & train
model = build_cnn_model()
model.summary()

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.1
)

# Evaluate
loss, acc = model.evaluate(test_images, test_labels)
print("CNN Test Accuracy:", acc)

model.save("cnn_cifar100_model.h5")
