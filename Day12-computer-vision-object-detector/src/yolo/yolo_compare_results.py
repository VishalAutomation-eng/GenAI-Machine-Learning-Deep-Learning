import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("data/images/zidane.jpg")
img2 = cv2.imread("runs/detect/predict/zidane.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img1)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(img2)
axes[1].set_title("YOLO Prediction")
axes[1].axis("off")

plt.show()
