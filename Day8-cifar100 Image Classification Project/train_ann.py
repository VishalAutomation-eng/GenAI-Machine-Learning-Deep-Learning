from data_loader import load_cifar100
from ann_model import build_ann_model

# Load data
train_images, train_labels, test_images, test_labels = load_cifar100()

# Flatten (same logic as Fashion-MNIST)
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

print("ANN Train shape:", train_images.shape)

# Build & train
model = build_ann_model()
model.summary()

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.1
)

# Evaluate
loss, acc = model.evaluate(test_images, test_labels)
print("ANN Test Accuracy:", acc)

model.save("ann_cifar100_model.h5")
