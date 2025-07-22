# prepare_validation_set.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 🧭 Dataset Configuration
DATASET_DIR = os.path.join("..", "cleaned_faces")  # Relative to model_training folder
CATEGORIES = ["mask", "no_mask"]
IMG_HEIGHT = 128  # ✅ Match model input height
IMG_WIDTH = 128   # ✅ Match model input width
OUTPUT_DIR = "model_training"

data, labels = [], []

# 🖼️ Load and Resize Images
for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    print(f"[📁] Looking for: {path}")

    if not os.path.exists(path):
        print(f"[⛔] Folder not found: {path}")
        continue

    label = CATEGORIES.index(category)
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        try:
            img = cv2.imread(fpath)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # 💡 Resize to (150, 128)
            data.append(img)
            labels.append(label)
        except:
            print(f"⚠️ Skipped unreadable image: {fpath}")

# 🧠 Prepare arrays
data = np.array(data) / 255.0  # Normalize
labels = to_categorical(np.array(labels))  # One-hot encode

# 🧪 Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# 💾 Save as .npy files
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

print(f"✅ Saved validation set to {OUTPUT_DIR}/ with image size: {IMG_HEIGHT}x{IMG_WIDTH}")