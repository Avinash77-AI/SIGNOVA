import cv2
import mediapipe as mp
import numpy as np
import csv

# ==========================================
# CHANGE ONLY THIS LABEL
# ==========================================

LABEL = "No"

DATA_FILE = "sign_data_normalized.csv"


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


print("--------------------------------")
print("   SIGNOVA NORMALIZED COLLECTOR")
print("--------------------------------")
print(f"Current Sign: {LABEL}")
print("Press S = Save sample")
print("Press Q = Quit")
print("--------------------------------")


sample_count = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame not received!")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    landmarks = []

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # ----------------------------------
        # Get wrist coordinates
        # ----------------------------------

        wrist = hand.landmark[0]

        wx = wrist.x
        wy = wrist.y
        wz = wrist.z

        normalized_points = []

        # ----------------------------------
        # Move wrist to origin
        # ----------------------------------

        for point in hand.landmark:

            x = point.x - wx
            y = point.y - wy
            z = point.z - wz

            normalized_points.append(
                [x, y, z]
            )

        # ----------------------------------
        # Scale normalization
        # ----------------------------------

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

        # ----------------------------------
        # Flatten
        # ----------------------------------

        for point in normalized_points:

            landmarks.extend(point)

        # ----------------------------------
        # Draw landmarks
        # ----------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"Sign: {LABEL}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "S = SAVE | Q = QUIT",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SIGNOVA - Normalized Data Collector",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("s"):

        if len(landmarks) == 63:

            with open(
                DATA_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [LABEL] + landmarks
                )

            sample_count += 1

            print(
                f"Sample saved: {sample_count}"
            )

        else:

            print(
                "No valid hand detected!"
            )


    elif key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()


print("--------------------------------")
print("Data collection stopped.")
print(f"Samples collected: {sample_count}")
print("--------------------------------")