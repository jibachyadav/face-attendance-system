"""
evaluate.py

Basic accuracy evaluation for the face recognition threshold-matching system.
This is NOT a trained-model accuracy metric (no training happens in this
project) -- it measures how well distance-based matching performs at the
current DISTANCE_THRESHOLD, using a small labeled test set.

Test image naming convention (place images in test_images/):
    <student_id>_<anything>.jpg   e.g. S001_alt.jpg, S001_lowlight.jpg
    unknown_<anything>.jpg        e.g. unknown_1.jpg  (a person NOT in the system)

Run:
    python recognition/evaluate.py
"""

import os
import sys
import pickle
import django
import numpy as np
from PIL import Image, ImageOps
import face_recognition

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

ENCODINGS_PATH = "models/encodings.pickle"
TEST_DIR = "test_images"
DISTANCE_THRESHOLD = 0.6


def load_image_corrected(path):
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def load_encodings():
    with open(ENCODINGS_PATH, "rb") as f:
        return pickle.load(f)


def evaluate():
    known_encodings, known_ids = load_encodings()

    if not os.path.isdir(TEST_DIR):
        print(f"Test folder '{TEST_DIR}/' not found. Create it and add labeled test images.")
        return

    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not test_files:
        print(f"No test images found in '{TEST_DIR}/'.")
        return

    results = []
    correct = 0
    false_accepts = 0
    false_rejects = 0
    correct_unknowns = 0

    for filename in sorted(test_files):
        expected_id = filename.split("_")[0]
        is_expected_unknown = expected_id.lower() == "unknown"
        path = os.path.join(TEST_DIR, filename)

        image = load_image_corrected(path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            results.append((filename, expected_id, "NO FACE DETECTED", "-"))
            continue

        encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(distances.argmin())
        best_distance = distances[best_idx]
        predicted_id = known_ids[best_idx] if best_distance < DISTANCE_THRESHOLD else "Unknown"

        if is_expected_unknown:
            if predicted_id == "Unknown":
                outcome = "CORRECT (unknown correctly rejected)"
                correct_unknowns += 1
            else:
                outcome = f"FALSE ACCEPT (matched to {predicted_id})"
                false_accepts += 1
        else:
            if predicted_id == expected_id:
                outcome = "CORRECT"
                correct += 1
            elif predicted_id == "Unknown":
                outcome = "FALSE REJECT (correct person not matched)"
                false_rejects += 1
            else:
                outcome = f"FALSE ACCEPT (matched to wrong person {predicted_id})"
                false_accepts += 1

        results.append((filename, expected_id, f"{predicted_id} (dist={best_distance:.3f})", outcome))

    print(f"\n{'Filename':<25} {'Expected':<12} {'Predicted':<25} Outcome")
    print("-" * 100)
    for filename, expected_id, predicted, outcome in results:
        print(f"{filename:<25} {expected_id:<12} {predicted:<25} {outcome}")

    total = len(results)
    print("\n--- Summary ---")
    print(f"Total test images:     {total}")
    print(f"Correct matches:       {correct}")
    print(f"Correct unknowns:      {correct_unknowns}")
    print(f"False accepts:         {false_accepts}  (wrong person matched)")
    print(f"False rejects:         {false_rejects}  (correct person not recognized)")
    print(f"Threshold used:        {DISTANCE_THRESHOLD}")


if __name__ == "__main__":
    evaluate()
