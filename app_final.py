import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import ttk
import threading
import pyttsx3
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"snapdragon_model\job_jpxlzw78p_optimized_onnx\model.onnx"
)

CLASSES = ["No", "Thank_You", "Yes", "hello"]

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LOAD SNAPDRAGON OPTIMIZED MODEL
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

print("==========================================")
print(" SIGNOVA - SNAPDRAGON OPTIMIZED MODEL")
print("==========================================")
print("Model loaded successfully")
print("Input :", INPUT_NAME)
print("Output:", OUTPUT_NAME)
print("Classes:", CLASSES)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# TEXT TO SPEECH
# ============================================================

tts = pyttsx3.init()


def speak(text):
    def run():
        tts.say(text)
        tts.runAndWait()

    threading.Thread(target=run, daemon=True).start()


# ============================================================
# NORMALIZE LANDMARKS
# SAME APPROACH USED DURING TRAINING
# ============================================================

def normalize_landmarks(hand_landmarks):

    points = []

    wrist = hand_landmarks.landmark[0]

    # Wrist-relative coordinates
    for landmark in hand_landmarks.landmark:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.extend([x, y, z])

    points = np.array(points, dtype=np.float32)

    # Scale normalization
    coords = points.reshape(21, 3)

    distances = np.linalg.norm(coords, axis=1)

    max_distance = np.max(distances)

    if max_distance > 0:
        points = points / max_distance

    return points.astype(np.float32)


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None
running = False

prediction_history = deque(maxlen=7)

current_prediction = "Waiting..."
current_confidence = 0.0


# ============================================================
# PREDICTION
# ============================================================

def predict_sign(features):

    x = features.reshape(1, 63).astype(np.float32)

    result = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: x}
    )

    probabilities = np.asarray(result[0])[0]

    index = int(np.argmax(probabilities))

    confidence = float(probabilities[index])

    predicted_class = CLASSES[index]

    return predicted_class, confidence, probabilities


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global camera
    global running
    global current_prediction
    global current_confidence

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Camera could not be opened")
        running = False
        return

    print("📷 Camera started")

    while running:

        success, frame = camera.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        detected_prediction = "No hand"
        detected_confidence = 0.0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            features = normalize_landmarks(hand)

            try:

                prediction, confidence, probabilities = (
                    predict_sign(features)
                )

                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_history.append(prediction)

                    # Majority vote for stability
                    counts = {
                        label: prediction_history.count(label)
                        for label in CLASSES
                    }

                    stable_prediction = max(
                        counts,
                        key=counts.get
                    )

                    detected_prediction = stable_prediction
                    detected_confidence = confidence

                else:
                    detected_prediction = "Uncertain"
                    detected_confidence = confidence

            except Exception as e:

                detected_prediction = "Inference Error"
                print("Inference error:", e)

        current_prediction = detected_prediction
        current_confidence = detected_confidence

        # Camera overlay
        cv2.rectangle(
            frame,
            (10, 10),
            (430, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Sign: {detected_prediction}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {detected_confidence * 100:.2f}%",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SIGNOVA - Snapdragon Optimized",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False

    if camera:
        camera.release()

    cv2.destroyAllWindows()

    print("📷 Camera stopped")


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "SIGNOVA - Snapdragon Optimized AI"
)

root.geometry("520x400")

root.resizable(False, False)


title = tk.Label(
    root,
    text="SIGNOVA",
    font=("Segoe UI", 24, "bold")
)

title.pack(pady=(25, 5))


subtitle = tk.Label(
    root,
    text="Snapdragon-Optimized Sign Recognition",
    font=("Segoe UI", 11)
)

subtitle.pack()


model_label = tk.Label(
    root,
    text="Model: Qualcomm AI Hub optimized ONNX",
    font=("Segoe UI", 10)
)

model_label.pack(pady=10)


prediction_label = tk.Label(
    root,
    text="Waiting...",
    font=("Segoe UI", 20, "bold")
)

prediction_label.pack(pady=20)


confidence_label = tk.Label(
    root,
    text="Confidence: 0.00%",
    font=("Segoe UI", 12)
)

confidence_label.pack()


def update_gui():

    prediction_label.config(
        text=current_prediction
    )

    confidence_label.config(
        text=f"Confidence: {current_confidence * 100:.2f}%"
    )

    root.after(100, update_gui)


def start_camera():

    global running

    if running:
        return

    running = True

    prediction_history.clear()

    threading.Thread(
        target=camera_loop,
        daemon=True
    ).start()


def stop_camera():

    global running

    running = False


def speak_current():

    if current_prediction in CLASSES:

        text = current_prediction.replace(
            "_",
            " "
        )

        speak(text)


button_frame = tk.Frame(root)

button_frame.pack(pady=25)


start_button = ttk.Button(
    button_frame,
    text="Start Camera",
    command=start_camera
)

start_button.grid(
    row=0,
    column=0,
    padx=8
)


stop_button = ttk.Button(
    button_frame,
    text="Stop Camera",
    command=stop_camera
)

stop_button.grid(
    row=0,
    column=1,
    padx=8
)


speak_button = ttk.Button(
    button_frame,
    text="Speak Sign",
    command=speak_current
)

speak_button.grid(
    row=0,
    column=2,
    padx=8
)


info = tk.Label(
    root,
    text="Press Q in the camera window to stop",
    font=("Segoe UI", 9)
)

info.pack(pady=5)


root.protocol(
    "WM_DELETE_WINDOW",
    lambda: (
        stop_camera(),
        root.destroy()
    )
)


update_gui()

root.mainloop()