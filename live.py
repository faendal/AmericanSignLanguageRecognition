from ultralytics import YOLO
import cv2
import time

model = YOLO("runs/detect/asl_model/weights/best.pt")
cap = cv2.VideoCapture(0)

last_letter = ""
last_time = 0
cooldown = 1.0
word = ""

print("[INFO] Presiona 'q' para salir - 'c' limpiar palabra - 's' mostrar palabra")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=416, conf=0.6)
    annotated = results[0].plot()

    if results and results[0].boxes and results[0].boxes.cls.numel() > 0:
        class_id = int(results[0].boxes.cls[0])
        label = model.names[class_id]

        current_time = time.time()
        if label != last_letter or (current_time - last_time) > cooldown:
            word += label
            last_letter = label
            last_time = current_time

        # Mostrar letra reconocida
        cv2.putText(
            annotated,
            f"Letra: {label}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    # Mostrar palabra en pantalla
    cv2.putText(
        annotated,
        f"Palabra: {word}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.imshow("Reconocimiento de Señales", annotated)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        word = ""
    elif key == ord("s"):
        print(f"[PALABRA FORMADA] {word}")

cap.release()
cv2.destroyAllWindows()
