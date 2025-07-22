import pyttsx3
import threading

engine = pyttsx3.init()
last_label_spoken = None

def speak(label):
    def run():
        try:
            engine.say(f"{label} detected")
            engine.runAndWait()
        except Exception as e:
            print(f"[❌] Voice error: {e}")
    t = threading.Thread(target=run)
    t.daemon = True  # ✅ Mark thread as safe for exit
    t.start()

def alert(label):
    global last_label_spoken
    label = label.strip().upper()
    if label != last_label_spoken and label in ["MASK", "NO MASK"]:
        speak(label)
        last_label_spoken = label