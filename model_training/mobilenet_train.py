# model_training/mobilenet_train.py

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# 📁 Directory setup
BASE_DIR = os.path.join("..","cleaned_faces")
MODEL_PATH = os.path.join("model_training", "mobilenet_model.h5")
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 📷 Image params
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# 🧠 Data pipeline
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# 🧠 Model architecture
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation="relu")(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 📊 Training & logging
print("[🧠] Training started...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save(MODEL_PATH)
print(f"[✅] Model saved at: {MODEL_PATH}")

# 📝 Save training log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"log_{timestamp}.txt")

with open(log_path, "w") as f:
    f.write(f"Training Timestamp: {timestamp}\n\n")
    for metric, values in history.history.items():
        f.write(f"{metric}: {values}\n")

print(f"[📊] Training log saved: {log_path}")