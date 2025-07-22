import os
import csv
from flask import Flask, render_template, send_from_directory
from flask import send_file

SNAPSHOT_DIR = "snapshots"
LOG_PATH = os.path.join(SNAPSHOT_DIR, "log.csv")

app = Flask(__name__)

def load_snapshot_log():
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(row)
    return sorted(logs, key=lambda x: x["timestamp"], reverse=True)

@app.route("/")
def index():
    entries = load_snapshot_log()
    return render_template("audit_view.html", entries=entries)

@app.route("/snapshots/<filename>")
def serve_image(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)
@app.route("/download_log")
def download_log():
    return send_file(LOG_PATH, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)