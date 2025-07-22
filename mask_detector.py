import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.overlay import draw_face_count, draw_timestamp
from utils.alerts import alert_no_mask, reset_alert

# Load model and Haar Cascade
model = load_model('model_training/model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 128
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    draw_face_count(frame, len(faces))  # Draw before loop

    for (x, y, w, h) in faces:
        # Prepare face crop
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        # Predict mask
        pred = model.predict(face_array)
        label = "Mask" if np.argmax(pred) == 0 else "No Mask"

        # Set box color and draw
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Voice alert
        if label == "No Mask":
            alert_no_mask()
        else:
            reset_alert()

    draw_timestamp(frame)  # Add timestamp after detection
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()