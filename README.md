# 🧠 Face Mask Detection using CNN and Flask

This project detects whether individuals are wearing face masks in real-time using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras, integrated with OpenCV and Flask for UI interaction.

## 📁 Project Structure

- `mask_detector.py`: Real-time detection using webcam
- `app.py`: Flask web app UI for uploading images or live stream
- `model.h5`: Trained CNN model
- `generate_report.py`: Script to generate summary stats and visuals
- `templates/`, `static/`: Front-end resources for Flask UI
- `utils/`: Reusable components for overlays, alerts, image handling
- `requirements.txt`: List of required Python packages

## 🚀 Setup

### 1. Python Environment
Ensure Python **3.11.x (stable release)** is installed.

```bash
python -m venv tf_env_final
.\tf_env_final\Scripts\activate