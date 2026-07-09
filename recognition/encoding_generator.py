"""
encoding_generator.py

Reads every reference photo in images/ (filename = student_id, e.g. S001.jpg),
detects the face, generates a 128-d face encoding, and saves everything to
models/encodings.pickle for the live recognition script to use.

Run this once whenever you add/remove reference photos:
    python recognition/encoding_generator.py
"""

import os
import pickle
import face_recognition

IMAGES_DIR = "images"
OUTPUT_PATH = "models/encodings.pickle"


def generate_encodings():
    known_encodings = []
    known_ids = []

    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No images found in '{IMAGES_DIR}/'. Add reference photos first.")
        return

    print(f"Found {len(image_files)} image(s). Generating encodings...")

    for filename in image_files:
        student_id = os.path.splitext(filename)[0]
        path = os.path.join(IMAGES_DIR, filename)

        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"  [SKIP] {filename}: no face detected")
            continue
        if len(face_locations) > 1:
            print(f"  [WARN] {filename}: {len(face_locations)} faces found, using the first one")

        encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
        known_encodings.append(encoding)
        known_ids.append(student_id)
        print(f"  [OK] {filename} -> {student_id}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((known_encodings, known_ids), f)

    print(f"\nSaved {len(known_ids)} encoding(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_encodings()
