from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    
    model.train(
        data="C:/Users/juanj/OneDrive - UPB/NoEstructurados/AmericanSignLanguageRecognition/asl_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="asl_model",
        device=0,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

