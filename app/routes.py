from flask import Blueprint, render_template, Response, jsonify, current_app
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import random
import time
import os
import threading

main_bp = Blueprint('main', __name__)

# Global variables
score = 0
target_letter = None
last_prediction_time = time.time()
prediction_delay = 3
prediction = "No Hand"
result_text = ""
TARGET_CLASSES = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LANDMARK_DIM = 126

# Thread safety
lock = threading.Lock()

# Model variables
model = None
label_encoder = None

# Initialize models function that we'll call from __init__.py
def init_models(app):
    global model, label_encoder, target_letter
    
    try:
        # Use source paths defined in config
        model = tf.keras.models.load_model(app.config['SOURCE_MODEL_PATH'])
        label_encoder = joblib.load(app.config['SOURCE_ENCODER_PATH'])
        target_letter = random.choice(TARGET_CLASSES)
        print("Model and encoder loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

def gen_frames():
    # Initialize camera and hands detector here to ensure they're in the right thread
    camera = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    
    global score, target_letter, last_prediction_time, prediction, result_text
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        local_prediction = "No Hand"
        
        if results.multi_hand_landmarks:
            landmarks_all = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmarks_all.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if len(landmarks_all) <= 2:
                if len(landmarks_all) == 1:
                    landmarks_all.append(np.zeros(63))  # Pad second hand with zeros
                input_data = np.concatenate(landmarks_all).reshape(1, -1)
                
                if input_data.shape[1] == LANDMARK_DIM:
                    try:
                        pred = model.predict(input_data, verbose=0)
                        class_index = np.argmax(pred)
                        local_prediction = label_encoder.inverse_transform([class_index])[0]
                        
                        # Update global prediction with thread safety
                        with lock:
                            prediction = local_prediction
                            
                            # Check match with target letter
                            current_time = time.time()
                            if current_time - last_prediction_time > prediction_delay:
                                if prediction == target_letter:
                                    score += 1
                                    result_text = "✅ Correct!"
                                else:
                                    score = max(score - 1, 0)
                                    result_text = f"❌ Wrong! You did '{prediction}'"
                                target_letter = random.choice(TARGET_CLASSES)
                                last_prediction_time = current_time
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")
                        local_prediction = "Error"
                else:
                    local_prediction = "Invalid Input"
        
        # Display UI elements
        with lock:
            current_score = score
            current_target = target_letter
            current_prediction = prediction
            current_result = result_text
            current_time = time.time()
        
        cv2.putText(frame, f"Target: {current_target}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
        cv2.putText(frame, f"Your Sign: {current_prediction}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        cv2.putText(frame, f"Score: {current_score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        
        if current_time - last_prediction_time < 1.0:
            cv2.putText(frame, current_result, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Encode frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@main_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/api/game_state')
def game_state():
    with lock:
        return jsonify({
            'score': score,
            'target_letter': target_letter,
            'prediction': prediction,
            'result_text': result_text,
            'time_remaining': max(0, prediction_delay - (time.time() - last_prediction_time))
        })