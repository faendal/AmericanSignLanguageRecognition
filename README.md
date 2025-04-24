# American Sign Language Recognition with YOLO v11

This project aims to implement a YOLOv11 based-model for recognizing American Sign Language (ASL) gestures.

## Dataset

The model is trained on a dataset of ASL letter signs and can be used for real-time sign language recognition.

The dataset can be found at [`American Sign Language Letters`](https://universe.roboflow.com/duyguj/american-sign-language-letters-vouo0/dataset/1)

## Model

For this project, a YOLOv11m model is used. It was trained with a NVIDIA GTX 1650 GPU and 4GB of VRAM, with CUDA 12.4.

The model was trained at [`Training`](https://github.com/faendal/AmericanSignLanguageRecognition/blob/main/train_model.py)

## Deployment

The model is only deployed locally, using a very simple interface made with `streamlit`.

The deployment code can be found at [`Deployment`](https://github.com/faendal/AmericanSignLanguageRecognition/blob/main/live.py)