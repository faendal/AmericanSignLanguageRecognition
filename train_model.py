from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")
    
    model.train(
        data="C:/Users/juanj/OneDrive - UPB/NoEstructurados/AmericanSignLanguageRecognition/asl_dataset/data.yaml",
        epochs=50,
        imgsz=416,
        batch=8,
        name="asl_model",
        patience=10,
        device=0,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
