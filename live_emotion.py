import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGE_SIZE = 64

# ==============================
# LOAD FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# LOAD EMOTION MODEL (INFERENCE MODE)
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
emotion_model = load_model(MODEL_PATH, compile=False)

# Warm-up model (IMPORTANT)
dummy_input = np.zeros((1, 64, 64, 1), dtype="float32")

emotion_model.predict(dummy_input, verbose=0)

print("✅ Model loaded and warmed up")

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Camera running. Press 'q' to quit.")
time.sleep(1)  # camera warm-up

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        try:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            preds = emotion_model.predict(face, verbose=0)
            emotion = EMOTION_LABELS[int(np.argmax(preds))]

            y_text = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(
                frame,
                emotion,
                (x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("⚠️ Emotion error:", e)

    cv2.imshow("Real-Time Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("👋 Exiting cleanly")
cap.release()
cv2.destroyAllWindows()
