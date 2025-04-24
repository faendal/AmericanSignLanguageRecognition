import streamlit as st
import cv2
import time
from ultralytics import YOLO

# Configuración inicial
st.set_page_config(page_title="Lenguaje de Señas", layout="wide")

# Cargar modelo
model = YOLO("runs/detect/asl_model2/weights/best.pt")

# Inicializar estado
if "running" not in st.session_state:
    st.session_state.running = False
if "word" not in st.session_state:
    st.session_state.word = ""
if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0
if "cooldown" not in st.session_state:
    st.session_state.cooldown = 1.0

# Sidebar
with st.sidebar:
    st.title("Reconocimiento de Señas")
    st.markdown("Usa tu cámara para detectar letras y formar palabras.")

    st.session_state.cooldown = st.slider(
        "Tiempo entre letras", 0.5, 2.5, st.session_state.cooldown, 0.1
    )

    if st.button("Iniciar"):
        st.session_state.running = True
    if st.button("Detener"):
        st.session_state.running = False
    if st.button("Limpiar palabra"):
        st.session_state.word = ""
        st.session_state.last_letter = ""
        st.session_state.last_time = 0
    if st.button("Guardar palabra"):
        with open("palabras_guardadas.txt", "a") as f:
            f.write(st.session_state.word + "\n")
        st.success("Palabra guardada.")

# Layout principal
st.title("Traductor Visual de Lenguaje de Señas")
frame_col, text_col = st.columns([3, 1])

frame_placeholder = frame_col.empty()
word_placeholder = text_col.empty()

# Bucle de cámara
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=416, conf=0.6)
        annotated = results[0].plot()

        if results and results[0].boxes and results[0].boxes.cls.numel() > 0:
            class_id = int(results[0].boxes.cls[0])
            label = model.names[class_id]

            current_time = time.time()
            if (
                label != st.session_state.last_letter
                or (current_time - st.session_state.last_time)
                > st.session_state.cooldown
            ):
                st.session_state.word += label
                st.session_state.last_letter = label
                st.session_state.last_time = current_time

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB")

        word_placeholder.markdown(
            f"""
            <div style='background-color:#1f77b4;padding:30px;border-radius:10px;color:white;text-align:center;font-size:30px;font-weight:bold;'>
            Palabra: {st.session_state.word}
            </div>
            """,
            unsafe_allow_html=True,
        )

    cap.release()
