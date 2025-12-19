import cv2
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model

# ==============================
# CONFIGURATION
# ==============================
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGE_SIZE = 64
CONFIDENCE_THRESHOLD = 0.6

emotion_window = deque(maxlen=10)

# ==============================
# LOAD FACE DETECTOR (STABLE)
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# LOAD EMOTION MODEL
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
emotion_model = load_model(MODEL_PATH, compile=False)

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Stable Emotion Recognition (press q to quit)")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve lighting robustness
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Tight face crop
        margin = int(0.15 * w)
        x1 = max(x + margin, 0)
        y1 = max(y + margin, 0)
        x2 = min(x + w - margin, gray.shape[1])
        y2 = min(y + h - margin, gray.shape[0])

        face = gray[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = emotion_model.predict(face, verbose=0)
        confidence = np.max(preds)
        emotion = EMOTION_LABELS[np.argmax(preds)]

        if confidence < CONFIDENCE_THRESHOLD:
            emotion = "Neutral"

        emotion_window.append(emotion)
        emotion = max(set(emotion_window), key=emotion_window.count)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10 if y > 20 else y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Application closed")
