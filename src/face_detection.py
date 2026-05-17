import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

# ── Configuration ─────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "../resources/known_faces")
DETECTION_CONFIDENCE = 0.6  # MediaPipe: min confidence to count as a face
PROCESS_EVERY_N_FRAMES = 40  # Skip frames for performance
SIMILARITY_THRESHOLD = 0.6  # DeepFace: 0.0-1.0, lower = stricter match
MODEL_NAME = "Facenet512"  # Options: Facenet, VGG-Face, ArcFace, DeepFace
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_full_range.tflite")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/1/"
    "blaze_face_full_range.tflite"
)


# ── Download MediaPipe model if needed ────────────────────────────────────────
def ensure_model(path, url):
    if not os.path.exists(path):
        print("Downloading face detection model...")
        urllib.request.urlretrieve(url, path)
        print(f"Model saved to '{path}'\n")


# ── Load known faces from folder ──────────────────────────────────────────────
def load_known_faces(directory):
    """
    Returns a list of (name, image_path) for every valid image in the folder.
    DeepFace handles its own encoding at match time.
    """
    if not os.path.exists(directory):
        print(f"Warning: '{directory}' not found. Creating it now.")
        os.makedirs(directory)
        print(f"Add face images to '{directory}/' and restart.\n")
        return []

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    known = []

    print("Loading known faces...")
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(supported):
            continue
        name = os.path.splitext(filename)[0]
        path = os.path.join(directory, filename)
        known.append((name, path))
        print(f"  Loaded: {name}")

    print(f"\nLoaded {len(known)} known face(s).\n")
    return known


def match_face(face_crop_bgr, known_faces):
    """
    Compare a cropped face (BGR numpy array) against all known faces.
    Returns (name, color) — green if matched, red if unknown.
    """
    for name, path in known_faces:
        try:
            result = DeepFace.verify(
                img1_path=face_crop_bgr,
                img2_path=path,
                model_name=MODEL_NAME,
                enforce_detection=False,  # we already cropped the face
                silent=True
            )
            distance = result["distance"]
            # print(f"  Comparing to {name}: distance={distance:.3f} (threshold={SIMILARITY_THRESHOLD})")
            if result["verified"] and distance < SIMILARITY_THRESHOLD:
                print(f"✅ Match found: {name} (distance={distance:.3f})")
                return name, (0, 255, 0)  # Green
        except Exception as e:
            print(f"  DeepFace error for {name}: {e}")
            continue

    return "Unknown", (0, 0, 255)  # Red


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model(MODEL_PATH, MODEL_URL)
    known_faces = load_known_faces(KNOWN_FACES_DIR)

    if not known_faces:
        print("No known faces loaded — all detections will show as Unknown.")

    # Build MediaPipe face detector
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    detector_options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=DETECTION_CONFIDENCE,
    )
    detector = mp_vision.FaceDetector.create_from_options(detector_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Webcam started. Press 'q' to quit.")
    print("  Green box = recognised face")
    print("  Red box   = unknown face\n")

    frame_count = 0
    last_results = []  # (x, y, w, h, name, color)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h_frame, w_frame = frame.shape[:2]

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = detector.detect(mp_image)

            last_results = []

            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                x2 = min(x + bbox.width, w_frame)
                y2 = min(y + bbox.height, h_frame)
                w = x2 - x
                h = y2 - y

                if w <= 0 or h <= 0:
                    continue

                # Crop the detected face for DeepFace
                face_crop = frame[y:y2, x:x2]

                if known_faces:
                    name, color = match_face(face_crop, known_faces)
                else:
                    name, color = "Unknown", (0, 0, 255)

                last_results.append((x, y, w, h, name, color))

        # Draw boxes on every frame using cached results
        for (x, y, w, h, name, color) in last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 22
            cv2.rectangle(frame, (x, label_y - 18), (x + w, label_y + 4), color, -1)
            cv2.putText(frame, name, (x + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        known_count = sum(1 for r in last_results if r[4] != "Unknown")
        unknown_count = len(last_results) - known_count
        cv2.putText(frame, f"Known: {known_count}  Unknown: {unknown_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
