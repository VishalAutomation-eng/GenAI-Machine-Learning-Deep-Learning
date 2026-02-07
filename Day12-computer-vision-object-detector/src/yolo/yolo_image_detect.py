from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.predict(
    source="data/images/zidane.jpg",
    save=True
)
