import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# 📝 Narrative text for report
project_text = {
    "Introduction": "This project aims to detect whether individuals are wearing face masks in real-time using a webcam, combining computer vision and deep learning to aid public safety.",
    "Abstract": "We used a Kaggle face mask dataset containing annotated images and XML files. After preprocessing and cropping faces, we built a classification system based on CNN architecture.",
    "Tools Used": "Python, OpenCV, TensorFlow/Keras, Haar Cascade Classifiers, Flask",
    "Steps Involved": [
        "Parsed XML annotations and cropped faces",
        "Organized dataset with mask/no_mask labels",
        "Trained CNN model using Keras",
        "Integrated OpenCV for live detection",
        "Built Flask frontend for deployment"
    ],
    "Conclusion": "The trained CNN accurately classifies masked and unmasked faces in real-time, providing a reliable alert system for environments requiring mask compliance."
}

# 🧾 PDF generator function
def generate_pdf(summary_lines, metrics):
    c = canvas.Canvas("face_mask_project_report.pdf", pagesize=A4)
    page_width, page_height = A4
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Page 1: Narrative and Flowchart ──
    y = 800
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Face Mask Detection with Live Alert System")
    c.setFont("Helvetica", 10)
    c.drawString(400, y, "Author: SRIJA DUTTA")
    c.drawString(50, y - 20, f"Generated on: {timestamp}")
    y -= 50

    c.setFont("Helvetica-Bold", 12)
    for section, content in project_text.items():
        c.drawString(50, y, section)
        y -= 20
        c.setFont("Helvetica", 9)
        if isinstance(content, list):
            for i, step in enumerate(content):
                c.drawString(60, y, f"{i+1}. {step}")
                y -= 15
        else:
            for line in content.split(". "):
                c.drawString(60, y, line.strip() + ".")
                y -= 15
        y -= 15
        c.setFont("Helvetica-Bold", 12)

    # 📌 Fixed flowchart position near bottom of page
    flowchart_path = "proj_flowchart.png"
    if os.path.exists(flowchart_path):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 220, "Project Flowchart:")
        c.drawImage(flowchart_path, 50, 40, width=500, height=160)

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "Page 1 • Narrative + Architecture by SRIJA DUTTA")
    c.showPage()

    # ── Page 2: Model Evaluation ──
    y = 800
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "📊 Model Evaluation & Metrics")
    c.setFont("Helvetica", 10)
    c.drawString(400, y, "Author: SRIJA DUTTA")
    y -= 30

    c.drawString(50, y, "Model Summary:")
    y -= 20
    c.setFont("Helvetica", 7)
    for line in summary_lines[:30]:
        c.drawString(50, y, line)
        y -= 10

    # 📊 Insert model layer bar plot
    model_plot_path = "model_layer_plot.png"
    if os.path.exists(model_plot_path):
        c.drawImage(model_plot_path, 50, y - 160, width=450, height=150)
        y -= 170

    c.setFont("Helvetica", 9)
    c.drawString(50, y, "Confusion Matrix:")
    y -= 20
    for row in metrics["confusion_matrix"]:
        c.drawString(80, y, str(row))
        y -= 15

    y -= 10
    c.drawString(50, y, "Classification Report:")
    y -= 20
    for label, stats in metrics["classification_report"].items():
        if isinstance(stats, dict):
            line = f"{label:12}  Prec: {stats['precision']:.2f}  Rec: {stats['recall']:.2f}  F1: {stats['f1-score']:.2f}"
            c.drawString(60, y, line)
            y -= 15

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "Page 2 • Technical Summary by SRIJA DUTTA")
    c.save()
    print("[✅] PDF report created cleanly: face_mask_project_report.pdf")

# 🚀 Script Entry Point
if __name__ == "__main__":
    model_path = os.path.join("model_training", "model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError("[⛔] Model not found.")
    model = load_model(model_path)
    print("[✅] Model loaded")

    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))

    # 🎨 Create bar chart of model layers
    layer_names = [layer.name for layer in model.layers]
    param_counts = [layer.count_params() for layer in model.layers]
    plt.figure(figsize=(10, 6))
    plt.barh(layer_names, param_counts, color="#0078D4")
    plt.title("Model Layers and Parameters", color="#333")
    plt.xlabel("Number of Parameters", fontsize=10)
    plt.tight_layout()
    plt.savefig("model_layer_plot.png")
    plt.close()
    print("[✅] Saved model_layer_plot.png")

    # 📦 Load validation data
    x_val_path = os.path.join("model_training", "X_val.npy")
    y_val_path = os.path.join("model_training", "y_val.npy")
    if os.path.exists(x_val_path) and os.path.exists(y_val_path):
        X_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        print("[✅] Loaded validation data")

        pred_classes = np.argmax(model.predict(X_val), axis=1)
        true_classes = np.argmax(y_val, axis=1)
        metrics = {
            "confusion_matrix": confusion_matrix(true_classes, pred_classes).tolist(),
            "classification_report": classification_report(true_classes, pred_classes, output_dict=True)
        }

        with open("evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        raise FileNotFoundError("[⛔] Validation data not found.")

    generate_pdf(summary_lines, metrics)