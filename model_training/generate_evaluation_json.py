# generate_evaluation_json.py

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# 📍 You are inside model_training folder when running this
model_path = "model.h5"
x_val_path = "X_val.npy"
y_val_path = "y_val.npy"
output_path = "../evaluation_metrics.json"  # Save metrics in root folder

# ✅ Load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"[⛔] Model file not found at: {model_path}")
model = load_model(model_path)
print("[✅] Model loaded")

# ✅ Load validation data
if not os.path.exists(x_val_path) or not os.path.exists(y_val_path):
    raise FileNotFoundError("[⛔] X_val.npy or y_val.npy missing.")
X_val = np.load(x_val_path)
y_val = np.load(y_val_path)
print(f"[📦] Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
print("[📐] Model expects input shape:", model.input_shape)

# 🔍 Confirm shape compatibility
if X_val.shape[1:] != model.input_shape[1:]:
    raise ValueError(f"[⛔] Shape mismatch: model expects {model.input_shape[1:]}, but got {X_val.shape[1:]}")

# 🔮 Predict and evaluate
pred_classes = np.argmax(model.predict(X_val), axis=1)
true_classes = np.argmax(y_val, axis=1)

metrics = {
    "confusion_matrix": confusion_matrix(true_classes, pred_classes).tolist(),
    "classification_report": classification_report(true_classes, pred_classes, output_dict=True)
}

# 💾 Save JSON in root folder
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)
print("[✅] evaluation_metrics.json created successfully.")