from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils.live_tools import overlay_prediction, save_snapshot
from utils.overlay import draw_face_count, draw_timestamp
from utils.alerts import alert
import os
import signal
import sys
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Paths
MODEL_PATH = os.path.join("model_training", "mobilenet_model.h5")
CASCADE_PATH = os.path.join("utils", "haarcascade_frontalface_default.xml")
SNAPSHOT_DIR = "snapshots"

# Load model and Haar cascade
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise IOError(f"[⛔] Haar cascade not found at {CASCADE_PATH}")
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# 🧠 Preprocess face image for model
def preprocess_face(face_img):
    face = cv2.resize(face_img, (128, 128))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    return np.expand_dims(face, axis=0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        draw_face_count(frame, len(faces))
        draw_timestamp(frame)

        for (x, y, w, h) in faces:
            if w < 80 or h < 80 or x < 5 or y < 5:
                continue

            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face = preprocess_face(face_img)
            pred = model.predict(face)[0]
            mask_score, no_mask_score = pred[0], pred[1]
            confidence = max(mask_score, no_mask_score)

            # 🧠 Assign label
            
            if mask_score >= 0.85:
                label = "MASK"
            elif no_mask_score >= 0.85:
                label = "NO MASK"
            else:
                label = "UNCERTAIN"

            # 🖼️ Overlay
            overlay_prediction(frame, x, y, w, h, label, mask_score, no_mask_score)

            # 🔊 Voice alert (threaded + cooldown)
            if label in ["MASK", "NO MASK"]:
                alert(label)
               
            # 💾 Snapshot saving and CSV logging
            save_snapshot(frame, label, confidence)

        # 📡 Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def signal_handler(sig, frame):
    print("🛑 Shutting down gracefully...")
    camera.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    app.run(debug=True)