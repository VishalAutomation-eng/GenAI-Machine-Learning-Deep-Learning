from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("data/images/bus.jpg")

for result in results:
    print("Boxes (xyxy):", result.boxes.xyxy)
    print("Confidence:", result.boxes.conf)
    print("Classes:", [
        result.names[int(cls)] for cls in result.boxes.cls
    ])
