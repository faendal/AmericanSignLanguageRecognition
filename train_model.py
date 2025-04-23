from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="C:/Users/juanj/OneDrive - UPB/NoEstructurados/AmericanSignLanguageRecognition/asl_dataset/data.yaml",
        epochs=30,
        imgsz=416,
        batch=4,
        name="asl_model",
        device=0,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

