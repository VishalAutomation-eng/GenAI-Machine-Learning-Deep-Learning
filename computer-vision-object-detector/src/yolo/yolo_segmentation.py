from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.predict(
    source="https://youtu.be/Qd6uWWd2Yoo",
    imgsz=320,
    save=True
)
