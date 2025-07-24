import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
from utils.overlay import draw_face_count, draw_timestamp
from utils.alerts import alert

# Load model and Haar Cascade
model = load_model('model_training/mobilenet_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 128
cap = cv2.VideoCapture(0)

label_buffer = deque(maxlen=10)  # Store last 10 predictions
last_announced_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    draw_face_count(frame, len(faces))

    final_label = None
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        pred = model.predict(face_array)[0]
        confidence = np.max(pred)
        label = "MASK" if np.argmax(pred) == 0 else "NO MASK"

        if confidence > 0.80:
            label_buffer.append(label)
        else:
            label_buffer.append("UNCERTAIN")

        # Use most frequent label in buffer as final output
        if len(label_buffer) == label_buffer.maxlen:
            label_counts = {l: label_buffer.count(l) for l in set(label_buffer)}
            final_label = max(label_counts, key=label_counts.get)
        else:
            final_label = label  # Early frames

        color = (0, 255, 0) if final_label == "MASK" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, final_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if final_label in ["MASK", "NO MASK"] and final_label != last_announced_label:
            alert(final_label)
            last_announced_label = final_label

    draw_timestamp(frame)
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()