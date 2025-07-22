from datetime import datetime
import cv2

def draw_face_count(frame, count):
    cv2.putText(frame, f"Faces Detected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

def draw_timestamp(frame):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)