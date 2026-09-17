from ultralytics import YOLO

# Load a lightweight YOLO model
model = YOLO("yolo11n.pt")

# Train on our GTSRB detection dataset
results = model.train(
    data="yolo_dataset/traffic_sign.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    project="models",
    name="traffic_sign_detector",
    patience=5
)

print("\n====================================")
print("YOLO TRAINING COMPLETE")
print("====================================")
print("Best model should be inside:")
print("models/traffic_sign_detector/weights/best.pt")
print("====================================")