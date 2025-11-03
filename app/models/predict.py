from flask import jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import random
import time
import os

# Use absolute paths based on project root to avoid path issues
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'isl_landmark_model.h5')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# Check if files exist and print paths for debugging
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder file not found at: {ENCODER_PATH}")

# Load model and encoder
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model and encoder loaded successfully")
except Exception as e:
    print(f"Error loading model or encoder: {str(e)}")
    raise

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

score = 0
target_letter = random.choice(TARGET_CLASSES)
last_prediction_time = time.time()
prediction_delay = 3  # seconds between predictions

def process_frame(frame):
    global score, target_letter, last_prediction_time

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No Hand"

    if results.multi_hand_landmarks:
        landmarks_all = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks_all.append(landmarks)

        if len(landmarks_all) <= 2:
            if len(landmarks_all) == 1:
                landmarks_all.append(np.zeros(63))  # Pad second hand with zeros
            input_data = np.concatenate(landmarks_all).reshape(1, -1)

            if input_data.shape[1] == LANDMARK_DIM:
                pred = model.predict(input_data, verbose=0)
                class_index = np.argmax(pred)
                prediction = label_encoder.inverse_transform([class_index])[0]

                if time.time() - last_prediction_time > prediction_delay:
                    if prediction == target_letter:
                        score += 1
                    else:
                        score = max(score - 1, 0)
                    target_letter = random.choice(TARGET_CLASSES)
                    last_prediction_time = time.time()
            else:
                prediction = "Invalid Input"

    return prediction, score, target_letter

def predict_gesture(landmarks_data):
    """
    Predicts the gesture from landmark data
    """
    if landmarks_data is None:
        return "No Hand"
    
    try:
        pred = model.predict(landmarks_data, verbose=0)
        class_index = np.argmax(pred)
        return label_encoder.inverse_transform([class_index])[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Prediction Error"