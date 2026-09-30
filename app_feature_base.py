import cv2
import os
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque, Counter
import pyttsx3
import threading
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from sign_dictionary import SIGN_GLOSS


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "signova_normalized_model.pkl"
VOSK_MODEL_PATH = "vosk-model"
VIDEO_FOLDER = "sign_videos"


# ============================================================
# LOAD SIGN RECOGNITION MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("SIGNOVA model loaded!")
print("Classes:", model.classes_)


# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()
engine.setProperty("rate", 150)


def speak_text(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(
        target=run_speech,
        daemon=True
    ).start()


# ============================================================
# LOAD VOSK OFFLINE SPEECH MODEL
# ============================================================

try:
    vosk_model = Model(VOSK_MODEL_PATH)
    print("Vosk offline speech model loaded!")

except Exception as e:
    vosk_model = None
    print("Vosk model error:", e)


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
# GLOBAL VARIABLES
# ============================================================

cap = None
running = False

prediction_history = deque(maxlen=10)

last_spoken = ""

voice_running = False
voice_thread = None


# ============================================================
# NORMALIZE HAND LANDMARKS
# ============================================================

def get_normalized_landmarks(hand):

    wrist = hand.landmark[0]

    wx = wrist.x
    wy = wrist.y
    wz = wrist.z

    points = []

    for point in hand.landmark:

        x = point.x - wx
        y = point.y - wy
        z = point.z - wz

        points.append([x, y, z])

    max_distance = 0

    for point in points:

        distance = np.sqrt(
            point[0] ** 2 +
            point[1] ** 2 +
            point[2] ** 2
        )

        max_distance = max(
            max_distance,
            distance
        )

    if max_distance > 0:

        for point in points:

            point[0] /= max_distance
            point[1] /= max_distance
            point[2] /= max_distance

    landmarks = []

    for point in points:
        landmarks.extend(point)

    return landmarks


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global cap
    global running

    if running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        status_label.config(
            text="CAMERA NOT FOUND"
        )

        return

    running = True

    status_label.config(
        text="● ONLINE"
    )

    update_camera()


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global cap
    global running

    running = False

    if cap is not None:

        cap.release()
        cap = None

    camera_label.config(
        image="",
        text="Camera stopped"
    )

    status_label.config(
        text="● OFFLINE"
    )


# ============================================================
# CAMERA UPDATE
# ============================================================

def update_camera():

    global last_spoken

    if not running:
        return

    success, frame = cap.read()

    if not success:

        root.after(
            30,
            update_camera
        )

        return

    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = get_normalized_landmarks(
            hand
        )

        if len(landmarks) == model.n_features_in_:

            input_data = np.array(
                landmarks
            ).reshape(
                1,
                -1
            )

            probabilities = model.predict_proba(
                input_data
            )[0]

            best_index = np.argmax(
                probabilities
            )

            prediction = model.classes_[
                best_index
            ]

            confidence = probabilities[
                best_index
            ]

            if confidence >= 0.50:

                prediction_history.append(
                    prediction
                )

            if len(prediction_history) > 0:

                most_common = Counter(
                    prediction_history
                ).most_common(1)[0]

                stable_prediction = most_common[0]

                stable_confidence = (
                    most_common[1] /
                    len(prediction_history)
                )

            else:

                stable_prediction = "Unknown"
                stable_confidence = 0

            detected_label.config(
                text=stable_prediction
                .replace("_", " ")
                .upper()
            )

            confidence_label.config(
                text=
                f"Confidence: "
                f"{stable_confidence * 100:.1f}%"
            )

            if (
                stable_prediction != last_spoken
                and stable_confidence >= 0.80
            ):

                speak_text(
                    stable_prediction
                    .replace("_", " ")
                )

                last_spoken = stable_prediction

    else:

        prediction_history.clear()

        detected_label.config(
            text="SHOW YOUR HAND"
        )

        confidence_label.config(
            text="Confidence: --"
        )

        last_spoken = ""

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        frame_rgb
    )

    image = image.resize(
        (720, 540)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_label.config(
        image=photo
    )

    camera_label.image = photo

    root.after(
        15,
        update_camera
    )


# ============================================================
# SPEAK CURRENT SIGN
# ============================================================

def speak_current():

    text = detected_label.cget(
        "text"
    )

    if text not in [
        "SHOW YOUR HAND",
        "UNKNOWN"
    ]:

        speak_text(text)


# ============================================================
# FIND SIGN VIDEO
# ============================================================

def get_video_path(text):

    text = text.lower().strip()

    # HELLO
    if "hello" in text:
        path = os.path.join(VIDEO_FOLDER, "hello.mp4")

        if os.path.exists(path):
            print("Hello video found:", path)
            return path

    # THANK YOU
    if "thank you" in text or "thankyou" in text:
        path = os.path.join(VIDEO_FOLDER, "thank_you.mp4")

        if os.path.exists(path):
            print("Thank You video found:", path)
            return path

    # YES
    if text == "yes" or " yes " in text:
        path = os.path.join(VIDEO_FOLDER, "yes.mp4")

        if os.path.exists(path):
            print("Yes video found:", path)
            return path

    # NO
    if text == "no" or " no " in text:
        path = os.path.join(VIDEO_FOLDER, "no.mp4")

        if os.path.exists(path):
            print("No video found:", path)
            return path

    return None


# ============================================================
# PLAY SIGN VIDEO
# ============================================================

def play_sign_video(video_path):

    print(
        "Playing video:",
        video_path
    )

    cap_video = cv2.VideoCapture(
        video_path
    )

    if not cap_video.isOpened():

        print(
            "ERROR: Could not open video"
        )

        sign_output_label.config(
            text="VIDEO ERROR",
            image=""
        )

        return

    def show_frame():

        success, frame = cap_video.read()

        if not success:

            cap_video.release()

            print(
                "Video finished."
            )

            return

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame
        )

        image.thumbnail(
            (260, 180)
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        sign_output_label.config(
            image=photo,
            text=""
        )

        sign_output_label.image = photo

        root.after(
            40,
            show_frame
        )

    show_frame()


# ============================================================
# SIGN DICTIONARY
# ============================================================

def get_sign_visual(text):

    text = text.lower().strip()

    for phrase, visual in SIGN_GLOSS.items():

        if phrase in text:

            return visual

    return "SIGN NOT AVAILABLE"


# ============================================================
# UPDATE VOICE TEXT + PLAY SIGN VIDEO
# ============================================================

def update_voice_text(text):

    print(
        "Recognized voice:",
        text
    )

    voice_text_label.config(
        text=text.upper()
    )

    video_path = get_video_path(
        text
    )

    if video_path:

        play_sign_video(
            video_path
        )

    else:

        sign_visual = get_sign_visual(
            text
        )

        sign_output_label.config(
            text=sign_visual,
            image=""
        )

        sign_output_label.image = None


# ============================================================
# OFFLINE VOICE RECOGNITION
# ============================================================

def voice_recognition():

    global voice_running

    if vosk_model is None:

        voice_status_label.config(
            text="Voice model not found"
        )

        return

    audio_queue = queue.Queue()

    recognizer = KaldiRecognizer(
        vosk_model,
        16000
    )

    def callback(
        indata,
        frames,
        time,
        status
    ):

        if voice_running:

            audio_queue.put(
                bytes(indata)
            )

    try:

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback
        ):

            while voice_running:

                data = audio_queue.get()

                if recognizer.AcceptWaveform(
                    data
                ):

                    result = json.loads(
                        recognizer.Result()
                    )

                    text = result.get(
                        "text",
                        ""
                    )

                    if text:

                        root.after(
                            0,
                            lambda t=text:
                            update_voice_text(t)
                        )

    except Exception as e:

        print(
            "Voice error:",
            e
        )

        root.after(
            0,
            lambda:
            voice_status_label.config(
                text="Microphone error"
            )
        )


# ============================================================
# START VOICE
# ============================================================

def start_voice():

    global voice_running
    global voice_thread

    if voice_running:
        return

    voice_running = True

    voice_status_label.config(
        text="● LISTENING"
    )

    voice_thread = threading.Thread(
        target=voice_recognition,
        daemon=True
    )

    voice_thread.start()


# ============================================================
# STOP VOICE
# ============================================================

def stop_voice():

    global voice_running

    voice_running = False

    voice_status_label.config(
        text="● VOICE OFF"
    )


# ============================================================
# CLEAR
# ============================================================

def clear_detection():

    global last_spoken

    prediction_history.clear()

    last_spoken = ""

    detected_label.config(
        text="SHOW YOUR HAND"
    )

    confidence_label.config(
        text="Confidence: --"
    )

    voice_text_label.config(
        text="VOICE TEXT"
    )

    sign_output_label.config(
        text="SIGN OUTPUT",
        image=""
    )

    sign_output_label.image = None


# ============================================================
# CLOSE APPLICATION
# ============================================================

def close_app():

    stop_camera()

    stop_voice()

    hands.close()

    root.destroy()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "SIGNOVA - Sign Language Communication"
)

root.geometry(
    "1100x800"
)

root.minsize(
    1000,
    750
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="SIGNOVA",
    font=("Arial", 28, "bold")
)

title_label.pack(
    pady=(15, 2)
)


subtitle_label = tk.Label(
    root,
    text="Bridging Signs. Connecting Voices.",
    font=("Arial", 12)
)

subtitle_label.pack(
    pady=(0, 15)
)


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = tk.Frame(root)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# ============================================================
# CAMERA FRAME
# ============================================================

camera_frame = tk.LabelFrame(
    main_frame,
    text="Live Camera",
    font=("Arial", 12, "bold")
)

camera_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


camera_label = tk.Label(
    camera_frame,
    text="Camera stopped",
    font=("Arial", 16)
)

camera_label.pack(
    padx=10,
    pady=10
)


# ============================================================
# RIGHT PANEL
# ============================================================

right_frame = tk.Frame(
    main_frame,
    width=300
)

right_frame.pack(
    side="right",
    fill="y",
    padx=(10, 0)
)


# ============================================================
# CAMERA STATUS
# ============================================================

status_label = tk.Label(
    right_frame,
    text="● OFFLINE",
    font=("Arial", 15, "bold")
)

status_label.pack(
    pady=15
)


# ============================================================
# DETECTED SIGN
# ============================================================

tk.Label(
    right_frame,
    text="Detected Sign",
    font=("Arial", 12)
).pack(
    pady=(10, 5)
)


detected_label = tk.Label(
    right_frame,
    text="SHOW YOUR HAND",
    font=("Arial", 22, "bold"),
    wraplength=280
)

detected_label.pack(
    pady=10
)


confidence_label = tk.Label(
    right_frame,
    text="Confidence: --",
    font=("Arial", 12)
)

confidence_label.pack(
    pady=10
)


# ============================================================
# CAMERA BUTTONS
# ============================================================

ttk.Button(
    right_frame,
    text="▶ Start Camera",
    command=start_camera
).pack(
    fill="x",
    padx=20,
    pady=5
)


ttk.Button(
    right_frame,
    text="■ Stop Camera",
    command=stop_camera
).pack(
    fill="x",
    padx=20,
    pady=5
)


ttk.Button(
    right_frame,
    text="🔊 Speak Sign",
    command=speak_current
).pack(
    fill="x",
    padx=20,
    pady=5
)


# ============================================================
# VOICE FRAME
# ============================================================

voice_frame = tk.LabelFrame(
    right_frame,
    text="Offline Voice",
    font=("Arial", 11, "bold")
)

voice_frame.pack(
    fill="x",
    padx=15,
    pady=15
)


voice_status_label = tk.Label(
    voice_frame,
    text="● VOICE OFF",
    font=("Arial", 11, "bold")
)

voice_status_label.pack(
    pady=5
)


voice_text_label = tk.Label(
    voice_frame,
    text="VOICE TEXT",
    font=("Arial", 16, "bold"),
    wraplength=260
)

voice_text_label.pack(
    pady=8
)


# ============================================================
# SIGN OUTPUT
# ============================================================

sign_output_label = tk.Label(
    voice_frame,
    text="SIGN OUTPUT",
    font=("Arial", 18, "bold"),
    wraplength=260
)

sign_output_label.pack(
    pady=8
)


# ============================================================
# VOICE BUTTONS
# ============================================================

ttk.Button(
    voice_frame,
    text="🎤 Start Voice",
    command=start_voice
).pack(
    fill="x",
    padx=10,
    pady=4
)


ttk.Button(
    voice_frame,
    text="■ Stop Voice",
    command=stop_voice
).pack(
    fill="x",
    padx=10,
    pady=4
)


# ============================================================
# CLEAR BUTTON
# ============================================================

ttk.Button(
    right_frame,
    text="Clear",
    command=clear_detection
).pack(
    fill="x",
    padx=20,
    pady=8
)


# ============================================================
# MODEL INFORMATION
# ============================================================

info_frame = tk.LabelFrame(
    right_frame,
    text="SIGNOVA Model",
    font=("Arial", 11, "bold")
)

info_frame.pack(
    fill="x",
    padx=15,
    pady=15
)


tk.Label(
    info_frame,
    text="Model: Random Forest"
).pack(
    anchor="w",
    padx=10,
    pady=3
)


tk.Label(
    info_frame,
    text="Features: 63"
).pack(
    anchor="w",
    padx=10,
    pady=3
)


tk.Label(
    info_frame,
    text="Signs: 4"
).pack(
    anchor="w",
    padx=10,
    pady=3
)


tk.Label(
    info_frame,
    text="Speech: Vosk Offline"
).pack(
    anchor="w",
    padx=10,
    pady=3
)


tk.Label(
    info_frame,
    text="Mode: Offline"
).pack(
    anchor="w",
    padx=10,
    pady=3
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="SIGNOVA • Privacy-first • Local AI",
    font=("Arial", 10)
)

footer.pack(
    pady=8
)


# ============================================================
# WINDOW CLOSE
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()