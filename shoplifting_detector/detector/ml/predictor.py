import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'video_pretrained.weights.h5')

TARGET_FRAMES = 32
IMG_SIZE = (128, 128)

def build_model():
    video_backbone = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=None,       # don't re-download imagenet weights
        pooling='avg'
    )
    video_backbone.trainable = False

    model = models.Sequential([
        layers.Input(shape=(TARGET_FRAMES, 128, 128, 3)),
        layers.TimeDistributed(video_backbone),
        layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load model once at startup
model = build_model()
model.load_weights(MODEL_WEIGHTS_PATH)

def load_and_sample_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame / 255.0)
    cap.release()

    if len(frames) == 0:
        return np.zeros((TARGET_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3))

    indices = np.linspace(0, len(frames) - 1, TARGET_FRAMES).astype(int)
    return np.array([frames[i] for i in indices])

def predict_video(video_path):
    frames = load_and_sample_video(video_path)
    input_tensor = np.expand_dims(frames, axis=0)
    prob = model.predict(input_tensor, verbose=0)[0][0]
    label = 'Shoplifting Detected' if prob > 0.5 else 'Normal'
    confidence = float(prob) if prob > 0.5 else float(1 - prob)
    return label, round(confidence * 100, 2)

print("=== Loading model from:", MODEL_WEIGHTS_PATH)
print("=== File exists:", os.path.exists(MODEL_WEIGHTS_PATH))