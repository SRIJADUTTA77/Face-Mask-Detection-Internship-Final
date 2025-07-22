# utils/live_tools.py

import cv2
import os
import time
import csv

# 📁 Snapshot directory (autocreate)
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# 🎨 Overlay prediction and scores
def overlay_prediction(frame, x, y, w, h, label, mask_score, no_mask_score):
    color = (0, 255, 0) if label == "MASK" else (0, 0, 255) if label == "NO MASK" else (255, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Mask: {mask_score:.2f} | NoMask: {no_mask_score:.2f}",
                (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 💾 Save snapshot & log to CSV
def save_snapshot(frame, label, confidence):
    if label == "NO MASK" and confidence > 0.85:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"[📸] Snapshot saved: {filename}")

        log_path = os.path.join(SNAPSHOT_DIR, "log.csv")
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["filename", "label", "confidence", "timestamp"])
            writer.writerow([filename, label, confidence, timestamp])