import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
import mediapipe as mp
import time

alert_start_time = None  # When alert started
ALERT_DURATION = 5       # seconds

# Load CNN model for eye detection
eye_model = tf.keras.models.load_model("new_eye_model.h5", compile=False)

# MediaPipe face mesh for MAR
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Haar face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Constants
MOUTH_IDX = [78, 308, 13, 14, 87, 317]  # approximate landmarks for MAR
YAWN_THRESH = 0.6
YAWN_FRAMES = 10
EYE_CLOSED_FRAMES_THRESH = 5

# State
yawn_counter = 0
yawns = 0
eye_closed_counter = 0
eye_status = "Unknown"
yawn_status = "No Yawn"

# Capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Eye detection via CNN
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        eye_roi_gray = gray[y:y+int(h/2), x:x+w]

        try:
            eye_resized = cv2.resize(eye_roi_gray, (224, 224))
            eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_GRAY2RGB)
            eye_input = eye_rgb.astype("float32") / 255.0
            eye_input = np.expand_dims(eye_input, axis=0)

            pred = eye_model.predict(eye_input, verbose=0)
            eye_status = "Closed" if pred[0][0] < 0.7 else "Open"

            if eye_status == "Closed":
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0

        except Exception as e:
            print("Eye detection error:", e)
            eye_status = "Unknown"
        break  # only process first face

    # Yawn detection via MAR
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w = frame.shape[:2]
            points = np.array([[int(face.landmark[i].x * w), int(face.landmark[i].y * h)] for i in MOUTH_IDX])
            A = dist.euclidean(points[2], points[3])
            B = dist.euclidean(points[4], points[5])
            mar = A / B

            if mar > YAWN_THRESH:
                yawn_counter += 1
            else:
                if yawn_counter >= YAWN_FRAMES:
                    yawns += 1
                    print("Yawn Detected!")
                    yawn_status = "Yawning"
                yawn_counter = 0
                yawn_status = "No Yawn"

            break  # only process first face

    # Drowsiness Alert Conditions
    current_time = time.time()
   # Check for new drowsiness alert trigger
    if eye_closed_counter > EYE_CLOSED_FRAMES_THRESH:
        alert_start_time = current_time

    if yawn_counter == YAWN_FRAMES:  # Trigger only once when yawn is confirmed
        alert_start_time = current_time

    # Display alert if within the alert window
    if alert_start_time and (current_time - alert_start_time) < ALERT_DURATION:
        cv2.putText(frame, "DROWSINESS ALERT!", (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display statuses
    cv2.putText(frame, f"Eye: {eye_status}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawn: {yawn_status}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawn Count: {yawns}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
