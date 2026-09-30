"""
SIGNOVA — Bridging Signs. Connecting Voices.
Premium UI build (app_snapdragon_premium.py)

Backend logic (landmark normalisation, ONNX prediction, majority-vote
smoothing, pyttsx3 TTS, Vosk offline voice, sign-video lookup) is carried over
from app_final_full.py. Only the UI, path handling, threading hygiene and
error handling were reworked.

Run:  python app_snapdragon_premium.py
"""

import json
import queue
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyttsx3
from PIL import Image, ImageTk

try:
    import sounddevice as sd
except Exception as _e:  # pragma: no cover
    sd = None
    print("sounddevice unavailable:", _e)

try:
    from vosk import Model, KaldiRecognizer
except Exception as _e:  # pragma: no cover
    Model = None
    KaldiRecognizer = None
    print("Vosk unavailable:", _e)

try:
    from sign_dictionary import SIGN_GLOSS
except Exception as _e:  # pragma: no cover
    SIGN_GLOSS = {}
    print("sign_dictionary unavailable:", _e)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "snapdragon_model" / "job_jpxlzw78p_optimized_onnx" / "model.onnx"
)
VOSK_MODEL_PATH = BASE_DIR / "vosk-model"
VIDEO_FOLDER = BASE_DIR / "sign_videos"

# Exact model output order — do not reorder.
CLASS_NAMES = ["No", "Thank_You", "Yes", "hello"]

DISPLAY_NAMES = {
    "No": "NO",
    "Thank_You": "THANK YOU",
    "Yes": "YES",
    "hello": "HELLO",
}

VIDEO_MAP = {
    "hello": "hello.mp4",
    "thank you": "thank_you.mp4",
    "thankyou": "thank_you.mp4",
    "yes": "yes.mp4",
    "no": "no.mp4",
}
REQUIRED_VIDEOS = ["hello.mp4", "thank_you.mp4", "yes.mp4", "no.mp4"]

VIDEO_PLAY_COUNT = 2  # times a sign clip plays before holding its last frame

HISTORY_LEN = 8
MIN_CONFIDENCE = 0.50   # frame must reach this to enter the vote history
SPEAK_VOTE_SHARE = 0.80  # majority share required before auto-speaking

# ---- Colors ----
BG = "#0B0F14"
CARD = "#121820"
CARD_2 = "#171E27"
TEXT = "#F4F7FA"
MUTED = "#8D99A6"
ACCENT = "#35D07F"
ACCENT_HOVER = "#4BE095"
ACCENT_DARK = "#1F9E5B"
BORDER = "#202A35"
VIEW_BG = "#080C10"
ERR = "#FF6B6B"
WARN = "#F5B942"

FONT = "Segoe UI"

# Values reported by Qualcomm AI Hub (not local measurements).
AI_HUB_PROFILE_TEXT = (
    "Qualcomm AI Hub Profile  \u00b7  Snapdragon X Elite CRD  \u00b7  "
    "NPU: 8 compute units  \u00b7  est. peak memory 1 MB  \u00b7  "
    "first app load 2244.4 ms  \u00b7  subsequent load 593.2 ms  \u00b7  "
    "inference 0.0 ms at displayed precision"
)
HOLDOUT_TEXT = "Holdout Test Accuracy: 100% (160 samples)"


# ============================================================
# 2. MODEL LOADING / SIGN RECOGNITION
# ============================================================

class SignClassifier:
    """Qualcomm AI Hub optimised ONNX classifier (local ONNX Runtime)."""

    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()
        self.avg_ms = None  # measured locally, smoothed

        print("SIGNOVA premium UI")
        print("Qualcomm optimized ONNX loaded")
        print("Input:", self.input_name)
        print("Output:", self.output_name)
        print("Providers:", self.providers)

    def predict(self, landmarks):
        """Returns (class_name, probability) or None for invalid output."""
        input_data = np.asarray(landmarks, dtype=np.float32).reshape(1, 63)

        t0 = time.perf_counter()
        result = self.session.run(
            [self.output_name], {self.input_name: input_data}
        )
        ms = (time.perf_counter() - t0) * 1000.0
        self.avg_ms = ms if self.avg_ms is None else 0.9 * self.avg_ms + 0.1 * ms

        probabilities = np.asarray(result[0]).reshape(-1)

        if probabilities.size != len(CLASS_NAMES) or not np.all(
            np.isfinite(probabilities)
        ):
            print("Invalid model output shape/values:", probabilities.shape)
            return None

        index = int(np.argmax(probabilities))
        return CLASS_NAMES[index], float(probabilities[index])


def describe_provider(providers):
    """Truthful description of the provider ONNX Runtime is really using."""
    first = providers[0] if providers else "unknown"
    if "QNN" in first:
        return "SNAPDRAGON NPU (QNN)", ACCENT, first
    if "Dml" in first:
        return "ONNX RUNTIME \u00b7 DIRECTML", ACCENT, first
    if "CPU" in first:
        return "ONNX RUNTIME \u00b7 CPU", ACCENT, first
    return f"ONNX RUNTIME \u00b7 {first}", ACCENT, first


def get_normalized_landmarks(hand_landmarks):
    """Wrist-relative, max-distance-scaled 63-feature vector (unchanged)."""
    points = []

    wrist = hand_landmarks.landmark[0]
    wx, wy, wz = wrist.x, wrist.y, wrist.z

    for point in hand_landmarks.landmark:
        points.append([point.x - wx, point.y - wy, point.z - wz])

    max_distance = 0
    for point in points:
        distance = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        max_distance = max(max_distance, distance)

    if max_distance > 0:
        for point in points:
            point[0] /= max_distance
            point[1] /= max_distance
            point[2] /= max_distance

    landmarks = []
    for point in points:
        landmarks.extend(point)
    return landmarks


def display_prediction(prediction):
    return DISPLAY_NAMES.get(prediction, prediction.replace("_", " ").upper())


# ============================================================
# 5. TEXT TO SPEECH
# ============================================================

_tts_lock = threading.Lock()


def speak_text(text):
    """Same pyttsx3 approach as app_final_full.py; calls are serialised."""

    def speak():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print("TTS error:", e)

    threading.Thread(target=speak, daemon=True).start()


# ============================================================
# 7. SIGN VIDEO LOOKUP
# ============================================================

def resolve_sign_key(text):
    """Exact phrase first (original behaviour), then phrase inside sentence."""
    text = text.lower().strip()
    if text in VIDEO_MAP:
        return text
    for phrase in sorted(VIDEO_MAP, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return phrase
    return None


def get_sign_visual(text):
    text = text.lower().strip()
    for phrase, visual in SIGN_GLOSS.items():
        if phrase in text:
            return visual
    return "SIGN NOT AVAILABLE"


def fit_size(w, h, max_w, max_h):
    scale = min(max_w / float(w), max_h / float(h))
    return max(1, int(w * scale)), max(1, int(h * scale))


class SignPlayer:
    """Plays one sign clip at a time inside a fixed-size box."""

    def __init__(self, root, box, label):
        self.root = root
        self.box = box
        self.label = label
        self.cap = None
        self.job = None
        self.photo = None
        self.plays_left = 0
        self.delay = 40

    def _clear_image(self):
        self.label.config(image="")
        self.photo = None

    def show_message(self, text, color=MUTED):
        self.stop()
        self._clear_image()
        self.label.config(text=text, fg=color)

    def stop(self):
        if self.job is not None:
            try:
                self.root.after_cancel(self.job)
            except Exception:
                pass
            self.job = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def play(self, path):
        self.stop()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            self.show_message("VIDEO ERROR", ERR)
            return
        self.cap = cap
        self.plays_left = VIDEO_PLAY_COUNT
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps) if 5 < fps < 120 else 40
        self.delay = max(15, self.delay)
        self._tick()

    def _tick(self):
        self.job = None
        if self.cap is None:
            return
        try:
            ok, frame = self.cap.read()
            if not ok:
                self.plays_left -= 1
                if self.plays_left > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
            if not ok:
                # finished: hold last frame, free the file handle
                self.cap.release()
                self.cap = None
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            bw = max(self.box.winfo_width() - 4, 50)
            bh = max(self.box.winfo_height() - 4, 50)
            nw, nh = fit_size(w, h, bw, bh)
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.label.config(image=photo, text="")
            self.photo = photo
            self.job = self.root.after(self.delay, self._tick)
        except Exception as e:
            print("Video playback error:", e)
            self.stop()


# ============================================================
# UI HELPERS
# ============================================================

def make_card(parent):
    return tk.Frame(
        parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )


def make_button(parent, text, command, primary=False):
    bg = ACCENT if primary else CARD_2
    hover = ACCENT_HOVER if primary else "#212B36"
    fg = "#04140B" if primary else TEXT
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=hover,
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=22,
        pady=10,
        font=(FONT, 10, "bold"),
        cursor="hand2",
        highlightthickness=1,
        highlightbackground=ACCENT if primary else BORDER,
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


class ConfidenceBar(tk.Canvas):
    def __init__(self, parent):
        super().__init__(
            parent, height=12, bg=CARD_2, highlightthickness=0, bd=0
        )
        self.value = 0.0
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, value):
        value = max(0.0, min(100.0, float(value)))
        if abs(value - self.value) < 0.005:
            return
        self.value = value
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        fill_w = int(w * self.value / 100.0)
        if fill_w > 0:
            self.create_rectangle(0, 0, fill_w, h, fill=ACCENT, width=0)


# ============================================================
# 9. APPLICATION
# ============================================================

class SignovaApp:
    def __init__(self, root):
        self.root = root
        self.closing = False

        # backend state
        self.classifier = None
        self.model_error = None
        self.hands = None
        self.hands_error = None
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # camera state
        self.cap = None
        self.running = False
        self.cam_job = None
        self.cam_fail = 0
        self.camera_photo = None
        self.history = deque(maxlen=HISTORY_LEN)  # (class_name, probability)
        self.last_spoken = ""
        self.current_sign = None  # display text of active detection
        self._det_cache = None
        self._last_runtime_ui = 0.0

        # voice state
        self.vosk_model = None
        self.vosk_state = "loading"  # loading | ready | missing | error
        self.voice_thread = None
        self.voice_event = None

        self.ui_queue = queue.Queue()
        self.poll_job = None

        self._build_ui()
        self.player = SignPlayer(root, self.video_box, self.video_label)

        self._init_backend()
        self._refresh_diagnostics()
        self._update_engine_ui()

        self.poll_job = root.after(50, self._poll_ui_queue)

    # --------------------------------------------------------
    # Backend init / diagnostics
    # --------------------------------------------------------

    def _init_backend(self):
        try:
            self.classifier = SignClassifier(MODEL_PATH)
        except Exception as e:
            self.model_error = str(e)
            print("Model error:", e)

        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            self.hands_error = str(e)
            print("MediaPipe error:", e)

        # Vosk model can be large: load off the UI thread.
        threading.Thread(target=self._load_vosk_worker, daemon=True).start()

    def _load_vosk_worker(self):
        if Model is None:
            self.ui_queue.put(("vosk", ("error", None)))
            return
        if not VOSK_MODEL_PATH.exists():
            self.ui_queue.put(("vosk", ("missing", None)))
            return
        try:
            model = Model(str(VOSK_MODEL_PATH))
            print("Vosk loaded")
            self.ui_queue.put(("vosk", ("ready", model)))
        except Exception as e:
            print("Vosk error:", e)
            self.ui_queue.put(("vosk", ("error", None)))

    def _refresh_diagnostics(self):
        model_ok = self.classifier is not None
        videos_found = sum(
            1 for v in REQUIRED_VIDEOS if (VIDEO_FOLDER / v).exists()
        )
        videos_ok = videos_found == len(REQUIRED_VIDEOS)

        def mark(ok):
            return "\u2713" if ok else "\u00d7"

        if self.vosk_state == "loading":
            vosk_txt = "\u2026 Voice model"
        else:
            vosk_txt = f"{mark(self.vosk_state == 'ready')} Voice model"

        self.diag_detail.config(
            text=(
                f"{mark(model_ok)} AI model     {vosk_txt}     "
                f"{mark(videos_ok)} Sign videos ({videos_found}/{len(REQUIRED_VIDEOS)})"
            )
        )

        if not model_ok:
            text, color = "\u25cf MODEL UNAVAILABLE", ERR
        elif self.vosk_state == "loading":
            text, color = "\u25cf STARTING\u2026", WARN
        elif self.vosk_state == "ready" and videos_ok and self.hands:
            text, color = "\u25cf SYSTEM READY", ACCENT
        else:
            text, color = "\u25cf READY \u00b7 LIMITED", WARN
        self.diag_label.config(text=text, fg=color)

    def _update_engine_ui(self):
        if self.classifier is None:
            self.engine_dot.config(fg=ERR)
            self.engine_text.config(text="MODEL NOT LOADED")
            self.strip_values["RUNTIME"].config(text="ONNX Runtime \u00b7 unavailable")
            provider = "not loaded"
        else:
            short, color, provider = describe_provider(self.classifier.providers)
            self.engine_dot.config(fg=color)
            self.engine_text.config(text=short)
            self.strip_values["RUNTIME"].config(text=short.title().replace("Onnx", "ONNX"))
        self._update_runtime_note(provider)

    def _update_runtime_note(self, provider=None):
        if provider is None and self.classifier is not None:
            provider = self.classifier.providers[0]
        ms = self.classifier.avg_ms if self.classifier else None
        ms_txt = f"inference avg {ms:.2f} ms (measured)" if ms is not None else "inference: not measured yet"
        self.note_local.config(
            text=(
                f"Local runtime (this PC)  \u00b7  {provider}  \u00b7  {ms_txt}"
                f"        {HOLDOUT_TEXT}"
            )
        )

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------

    def _build_ui(self):
        root = self.root
        root.title("SIGNOVA \u2014 Bridging Signs. Connecting Voices.")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(1280, sw - 40)
        h = min(800, sh - 80)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(1000, 640)
        root.configure(bg=BG)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_main()
        self._build_controls()
        self._build_engine_strip()

        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(16, 8))
        header.columnconfigure(1, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(
            left, text="SIGNOVA", bg=BG, fg=TEXT, font=(FONT, 24, "bold")
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Bridging Signs. Connecting Voices.",
            bg=BG,
            fg=MUTED,
            font=(FONT, 10),
        ).pack(anchor="w")

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=2, sticky="e")

        badge = tk.Frame(right, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        badge.pack(anchor="e")
        tk.Label(
            badge, text="AI ENGINE", bg=CARD, fg=MUTED, font=(FONT, 9, "bold")
        ).pack(side="left", padx=(12, 6), pady=6)
        self.engine_dot = tk.Label(
            badge, text="\u25cf", bg=CARD, fg=MUTED, font=(FONT, 10)
        )
        self.engine_dot.pack(side="left")
        self.engine_text = tk.Label(
            badge, text="LOADING", bg=CARD, fg=TEXT, font=(FONT, 9, "bold")
        )
        self.engine_text.pack(side="left", padx=(4, 12))

        diag = tk.Frame(right, bg=BG)
        diag.pack(anchor="e", pady=(6, 0))
        self.diag_label = tk.Label(
            diag, text="\u25cf STARTING\u2026", bg=BG, fg=WARN, font=(FONT, 9, "bold")
        )
        self.diag_label.pack(side="left")
        self.diag_detail = tk.Label(
            diag, text="", bg=BG, fg=MUTED, font=(FONT, 8)
        )
        self.diag_detail.pack(side="left", padx=(12, 0))

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=1, column=0, sticky="nsew", padx=28, pady=6)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2, minsize=400)
        main.rowconfigure(0, weight=1)

        # ---------------- camera card ----------------
        cam_card = make_card(main)
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        cam_card.columnconfigure(0, weight=1)
        cam_card.rowconfigure(1, weight=1)

        cam_head = tk.Frame(cam_card, bg=CARD)
        cam_head.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
        tk.Label(
            cam_head, text="LIVE CAMERA", bg=CARD, fg=TEXT, font=(FONT, 11, "bold")
        ).pack(side="left")
        self.camera_status = tk.Label(
            cam_head, text="\u25cf CAMERA READY", bg=CARD, fg=MUTED,
            font=(FONT, 9, "bold"),
        )
        self.camera_status.pack(side="right")

        self.cam_box = tk.Frame(
            cam_card, bg=VIEW_BG, width=480, height=300,
            highlightbackground=BORDER, highlightthickness=1,
        )
        self.cam_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.cam_box.pack_propagate(False)

        self.camera_label = tk.Label(
            self.cam_box,
            text="Press START CAMERA",
            bg=VIEW_BG, fg=MUTED, font=(FONT, 14),
        )
        self.camera_label.pack(expand=True)

        # ---------------- right column ----------------
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # detection card
        det = make_card(right)
        det.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            det, text="DETECTED SIGN", bg=CARD, fg=MUTED, font=(FONT, 9, "bold")
        ).pack(anchor="w", padx=18, pady=(14, 0))

        self.detected_value = tk.Label(
            det, text="START THE CAMERA", bg=CARD, fg=MUTED,
            font=(FONT, 28, "bold"), wraplength=340, justify="left", anchor="w",
        )
        self.detected_value.pack(fill="x", padx=18, pady=(6, 8))

        conf_row = tk.Frame(det, bg=CARD)
        conf_row.pack(fill="x", padx=18)
        tk.Label(
            conf_row, text="Confidence", bg=CARD, fg=MUTED, font=(FONT, 10)
        ).pack(side="left")
        self.confidence_value = tk.Label(
            conf_row, text="--", bg=CARD, fg=ACCENT, font=(FONT, 13, "bold")
        )
        self.confidence_value.pack(side="right")

        self.confidence_bar = ConfidenceBar(det)
        self.confidence_bar.pack(fill="x", padx=18, pady=(8, 18))

        # voice -> sign card
        voice = make_card(right)
        voice.grid(row=1, column=0, sticky="nsew")
        voice.columnconfigure(0, weight=1)
        voice.rowconfigure(2, weight=1)

        v_head = tk.Frame(voice, bg=CARD)
        v_head.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 0))
        tk.Label(
            v_head, text="VOICE \u2192 SIGN", bg=CARD, fg=MUTED, font=(FONT, 9, "bold")
        ).pack(side="left")
        self.voice_status = tk.Label(
            v_head, text="VOICE OFF", bg=CARD, fg=MUTED, font=(FONT, 9, "bold")
        )
        self.voice_status.pack(side="right")

        self.voice_value = tk.Label(
            voice, text="SAY A SIGN WORD", bg=CARD, fg=MUTED,
            font=(FONT, 18, "bold"), wraplength=340, justify="left", anchor="w",
        )
        self.voice_value.grid(row=1, column=0, sticky="ew", padx=18, pady=(6, 8))

        self.video_box = tk.Frame(
            voice, bg=VIEW_BG, width=200, height=150,
            highlightbackground=BORDER, highlightthickness=1,
        )
        self.video_box.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.video_box.pack_propagate(False)
        self.video_label = tk.Label(
            self.video_box, text="SIGN VIDEO", bg=VIEW_BG, fg=MUTED,
            font=(FONT, 12, "bold"), justify="center",
        )
        self.video_label.pack(expand=True)

    def _build_controls(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.grid(row=2, column=0, sticky="ew", padx=28, pady=(8, 8))

        make_button(bar, "START CAMERA", self.start_camera, primary=True).pack(
            side="left", padx=(0, 10)
        )
        make_button(bar, "STOP", self.stop_camera).pack(side="left", padx=(0, 10))
        make_button(bar, "SPEAK SIGN", self.speak_current).pack(side="left", padx=(0, 10))
        self.voice_btn = make_button(bar, "VOICE", self.toggle_voice)
        self.voice_btn.pack(side="left", padx=(0, 10))
        make_button(bar, "CLEAR", self.clear_all).pack(side="right")

    def _build_engine_strip(self):
        strip = make_card(self.root)
        strip.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 16))

        fields = tk.Frame(strip, bg=CARD)
        fields.pack(fill="x", padx=18, pady=(10, 4))

        items = [
            ("AI ENGINE", "Qualcomm AI Hub Optimized"),
            ("MODEL", "SIGNOVA Neural Network"),
            ("RUNTIME", "ONNX Runtime"),
            ("MODE", "Offline \u00b7 Local Inference"),
            ("TARGET", "Snapdragon X Series"),
        ]
        self.strip_values = {}
        for i, (title, value) in enumerate(items):
            fields.columnconfigure(i, weight=1, uniform="strip")
            cell = tk.Frame(fields, bg=CARD)
            cell.grid(row=0, column=i, sticky="w")
            tk.Label(cell, text=title, bg=CARD, fg=MUTED, font=(FONT, 8, "bold")).pack(anchor="w")
            lbl = tk.Label(cell, text=value, bg=CARD, fg=TEXT, font=(FONT, 10, "bold"))
            lbl.pack(anchor="w")
            self.strip_values[title] = lbl

        notes = tk.Frame(strip, bg=CARD)
        notes.pack(fill="x", padx=18, pady=(2, 10))
        self.note_profile = tk.Label(
            notes, text=AI_HUB_PROFILE_TEXT, bg=CARD, fg=MUTED,
            font=(FONT, 8), justify="left", anchor="w",
        )
        self.note_profile.pack(fill="x")
        self.note_local = tk.Label(
            notes, text="", bg=CARD, fg=MUTED,
            font=(FONT, 8), justify="left", anchor="w",
        )
        self.note_local.pack(fill="x")
        notes.bind(
            "<Configure>",
            lambda e: (
                self.note_profile.config(wraplength=max(200, e.width)),
                self.note_local.config(wraplength=max(200, e.width)),
            ),
        )

    # --------------------------------------------------------
    # Detection display
    # --------------------------------------------------------

    def _idle_text(self):
        return "SHOW YOUR HAND" if self.running else "START THE CAMERA"

    def set_detection(self, text, conf_pct, active):
        state = (text, None if conf_pct is None else round(conf_pct, 2), active)
        if state == self._det_cache:
            return
        self._det_cache = state
        self.current_sign = text if active else None
        self.detected_value.config(text=text, fg=TEXT if active else MUTED)
        if conf_pct is None:
            self.confidence_value.config(text="--")
            self.confidence_bar.set(0)
        else:
            self.confidence_value.config(text=f"{conf_pct:.2f}%")
            self.confidence_bar.set(conf_pct)

    def set_idle_detection(self):
        self.set_detection(self._idle_text(), None, False)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    def _set_camera_status(self, text, color):
        self.camera_status.config(text=text, fg=color)

    def start_camera(self):
        if self.running or self.closing:
            return
        if self.classifier is None:
            self._set_camera_status("\u25cf MODEL UNAVAILABLE", ERR)
            return
        if self.hands is None:
            self._set_camera_status("\u25cf HAND TRACKING UNAVAILABLE", ERR)
            return

        try:
            cap = cv2.VideoCapture(0)
        except Exception as e:
            print("Camera error:", e)
            cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self._set_camera_status("\u25cf CAMERA UNAVAILABLE", ERR)
            self.camera_label.config(image="", text="Camera unavailable", fg=ERR)
            self.camera_photo = None
            return

        self.cap = cap
        self.running = True
        self.cam_fail = 0
        self.history.clear()
        self.last_spoken = ""
        self.cam_box.config(highlightbackground=ACCENT)
        self._set_camera_status("\u25cf CAMERA ACTIVE", ACCENT)
        self.set_idle_detection()
        self._camera_tick()

    def stop_camera(self, status_text="\u25cf CAMERA READY", status_color=MUTED):
        self.running = False
        if self.cam_job is not None:
            try:
                self.root.after_cancel(self.cam_job)
            except Exception:
                pass
            self.cam_job = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.history.clear()
        self.last_spoken = ""

        if not self.closing:
            self.cam_box.config(highlightbackground=BORDER)
            self._set_camera_status(status_text, status_color)
            self.camera_label.config(image="", text="Camera stopped", fg=MUTED)
            self.camera_photo = None
            self.set_idle_detection()

    def _camera_tick(self):
        self.cam_job = None
        if not self.running or self.cap is None or self.closing:
            return

        try:
            ok, frame = self.cap.read()
        except Exception as e:
            print("Camera read error:", e)
            ok, frame = False, None

        if not ok:
            self.cam_fail += 1
            if self.cam_fail >= 60:
                self.stop_camera("\u25cf CAMERA STREAM LOST", ERR)
                return
            self.cam_job = self.root.after(30, self._camera_tick)
            return
        self.cam_fail = 0

        try:
            frame = self._process_frame(frame)
            self._show_camera_frame(frame)
        except Exception as e:
            print("Frame error:", e)

        if self.running and not self.closing:
            self.cam_job = self.root.after(15, self._camera_tick)

    def _process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            landmarks = get_normalized_landmarks(hand)
            if len(landmarks) == 63:
                prediction = self.classifier.predict(landmarks)
                if prediction is not None:
                    name, confidence = prediction
                    if confidence >= MIN_CONFIDENCE:
                        self.history.append((name, confidence))

                if self.history:
                    votes = Counter(n for n, _ in self.history)
                    stable_name, stable_count = votes.most_common(1)[0]
                    vote_share = stable_count / len(self.history)
                    # displayed confidence = real model probability, averaged
                    # over the frames that voted for the stable label
                    probs = [c for n, c in self.history if n == stable_name]
                    model_conf = 100.0 * sum(probs) / len(probs)

                    shown = display_prediction(stable_name)
                    self.set_detection(shown, model_conf, True)

                    if stable_name != self.last_spoken and vote_share >= SPEAK_VOTE_SHARE:
                        speak_text(shown)
                        self.last_spoken = stable_name
                else:
                    self.set_detection("UNKNOWN", None, False)
        else:
            self.history.clear()
            self.last_spoken = ""
            self.set_idle_detection()

        now = time.time()
        if now - self._last_runtime_ui > 1.0:
            self._last_runtime_ui = now
            self._update_runtime_note()

        return frame

    def _show_camera_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bw = self.cam_box.winfo_width() - 4
        bh = self.cam_box.winfo_height() - 4
        if bw < 50 or bh < 50:
            bw, bh = 640, 480
        nw, nh = fit_size(w, h, bw, bh)
        interp = cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.camera_label.config(image=photo, text="")
        self.camera_photo = photo

    # --------------------------------------------------------
    # Speak sign
    # --------------------------------------------------------

    def speak_current(self):
        if self.current_sign:
            speak_text(self.current_sign)

    # --------------------------------------------------------
    # Voice -> Sign
    # --------------------------------------------------------

    def _set_voice_status(self, text, color):
        self.voice_status.config(text=text, fg=color)

    def _set_voice_button(self, active):
        if active:
            self.voice_btn.config(
                text="STOP VOICE", fg=ACCENT, highlightbackground=ACCENT
            )
        else:
            self.voice_btn.config(
                text="VOICE", fg=TEXT, highlightbackground=BORDER
            )

    def toggle_voice(self):
        if self.voice_event is not None and not self.voice_event.is_set():
            self.stop_voice()
        else:
            self.start_voice()

    def start_voice(self):
        if self.closing:
            return
        if self.voice_event is not None and not self.voice_event.is_set():
            return
        if self.vosk_state == "loading":
            self._set_voice_status("VOICE MODEL LOADING\u2026", WARN)
            return
        if self.vosk_model is None:
            self._set_voice_status("VOICE MODEL NOT FOUND", ERR)
            return
        if sd is None or KaldiRecognizer is None:
            self._set_voice_status("MICROPHONE UNAVAILABLE", ERR)
            return

        event = threading.Event()
        self.voice_event = event
        self._set_voice_status("\u25cf STARTING\u2026", WARN)
        self._set_voice_button(True)

        self.voice_thread = threading.Thread(
            target=self._voice_worker, args=(event,), daemon=True
        )
        self.voice_thread.start()

    def stop_voice(self):
        if self.voice_event is not None:
            self.voice_event.set()
        self._set_voice_button(False)
        self._set_voice_status("VOICE OFF", MUTED)

    def _voice_worker(self, event):
        """Background thread: no Tk calls, results go through ui_queue."""
        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if not event.is_set():
                audio_queue.put(bytes(indata))

        try:
            recognizer = KaldiRecognizer(self.vosk_model, 16000)
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                self.ui_queue.put(("voice_state", (event, "listening")))
                while not event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text:
                            self.ui_queue.put(("voice_text", (event, text)))
        except Exception as e:
            print("Voice error:", e)
            self.ui_queue.put(("voice_error", (event, "MICROPHONE UNAVAILABLE")))
        finally:
            self.ui_queue.put(("voice_ended", event))

    def update_voice_text(self, text):
        self.voice_value.config(text=text.upper(), fg=TEXT)

        key = resolve_sign_key(text)
        if key:
            filename = VIDEO_MAP[key]
            path = VIDEO_FOLDER / filename
            if path.exists():
                self.player.play(path)
            else:
                print("Sign video missing:", path)
                self.player.show_message(f"SIGN VIDEO MISSING\n{filename}", WARN)
        else:
            self.player.show_message(get_sign_visual(text), MUTED)

    # --------------------------------------------------------
    # Thread -> UI messages
    # --------------------------------------------------------

    def _poll_ui_queue(self):
        self.poll_job = None
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                try:
                    self._handle_message(kind, payload)
                except Exception as e:
                    print("UI message error:", e)
        except queue.Empty:
            pass
        if not self.closing:
            self.poll_job = self.root.after(50, self._poll_ui_queue)

    def _is_current_voice(self, event):
        return event is self.voice_event and not event.is_set()

    def _handle_message(self, kind, payload):
        if kind == "vosk":
            state, model = payload
            self.vosk_state = state
            self.vosk_model = model
            self._refresh_diagnostics()
        elif kind == "voice_state":
            event, state = payload
            if self._is_current_voice(event) and state == "listening":
                self._set_voice_status("\u25cf LISTENING", ACCENT)
        elif kind == "voice_text":
            event, text = payload
            if self._is_current_voice(event):
                self.update_voice_text(text)
        elif kind == "voice_error":
            event, message = payload
            if event is self.voice_event:
                event.set()
                self._set_voice_button(False)
                self._set_voice_status(message, ERR)
        elif kind == "voice_ended":
            if payload is self.voice_event and not payload.is_set():
                payload.set()
                self._set_voice_button(False)

    # --------------------------------------------------------
    # Clear
    # --------------------------------------------------------

    def clear_all(self):
        self.history.clear()
        self.last_spoken = ""
        self._det_cache = None
        self.set_idle_detection()

        self.voice_value.config(text="SAY A SIGN WORD", fg=MUTED)
        self.player.show_message("SIGN VIDEO", MUTED)

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    def close(self):
        if self.closing:
            return
        self.closing = True

        for job in (self.poll_job,):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
        self.poll_job = None

        try:
            self.stop_camera()
        except Exception as e:
            print("Close (camera):", e)

        try:
            if self.voice_event is not None:
                self.voice_event.set()
            if self.voice_thread is not None and self.voice_thread.is_alive():
                self.voice_thread.join(timeout=1.5)
        except Exception as e:
            print("Close (voice):", e)

        try:
            self.player.stop()
        except Exception as e:
            print("Close (video):", e)

        try:
            if self.hands is not None:
                self.hands.close()
        except Exception as e:
            print("Close (mediapipe):", e)

        try:
            self.root.destroy()
        except Exception:
            pass


# ============================================================
# RUN
# ============================================================

def main():
    root = tk.Tk()
    SignovaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()