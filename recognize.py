import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_FILE = "signova_normalized_model.pkl"

model = joblib.load(MODEL_FILE)

print("================================")
print("       SIGNOVA")
print("================================")
print("Normalized model loaded!")
print("Classes:", model.classes_)
print("================================")


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()


# ==========================================
# SMOOTHING
# ==========================================

prediction_history = deque(maxlen=10)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    landmarks = []


    # ======================================
    # HAND DETECTED
    # ======================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # -------------------------------
        # Wrist
        # -------------------------------

        wrist = hand.landmark[0]

        wx = wrist.x
        wy = wrist.y
        wz = wrist.z


        # -------------------------------
        # Normalize landmarks
        # -------------------------------

        normalized_points = []

        for point in hand.landmark:

            x = point.x - wx
            y = point.y - wy
            z = point.z - wz

            normalized_points.append(
                [x, y, z]
            )


        # -------------------------------
        # Scale normalization
        # -------------------------------

        max_distance = 0

        for point in normalized_points:

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

            for point in normalized_points:

                point[0] /= max_distance
                point[1] /= max_distance
                point[2] /= max_distance


        # -------------------------------
        # Flatten
        # -------------------------------

        for point in normalized_points:

            landmarks.extend(point)


        # -------------------------------
        # Draw hand
        # -------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # ==================================
        # PREDICTION
        # ==================================

        if len(landmarks) == model.n_features_in_:

            input_data = np.array(
                landmarks
            ).reshape(1, -1)


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


            # -------------------------------
            # Add prediction to history
            # -------------------------------

            if confidence >= 0.50:

                prediction_history.append(
                    prediction
                )


            # -------------------------------
            # Stable prediction
            # -------------------------------

            if len(prediction_history) > 0:

                most_common = Counter(
                    prediction_history
                ).most_common(1)[0]

                stable_prediction = (
                    most_common[0]
                )

                stable_count = (
                    most_common[1]
                )

                stable_confidence = (
                    stable_count /
                    len(prediction_history)
                )

            else:

                stable_prediction = "Unknown"
                stable_confidence = 0


            # ==================================
            # DISPLAY
            # ==================================

            cv2.putText(
                frame,
                f"Detected: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: "
                f"{stable_confidence * 100:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    # ======================================
    # NO HAND
    # ======================================

    else:

        prediction_history.clear()

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


    # ======================================
    # TITLE
    # ======================================

    cv2.putText(
        frame,
        "SIGNOVA - LIVE RECOGNITION",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "SIGNOVA",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("SIGNOVA stopped.")