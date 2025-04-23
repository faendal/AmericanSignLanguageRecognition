from ultralytics import YOLO

# Cargar modelo base (pequeño para tu GPU)
model = YOLO("yolov8n.pt")

# Entrenar
model.train(
    data="asl_dataset/data.yaml",
    epochs=30,
    imgsz=416,
    batch=4,
    name="asl_model",
    device=0,
)
