"""
encoding_generator.py

Reads every Student record that has a photo uploaded (via Django admin),
corrects image orientation based on EXIF data (phone photos often store
rotation as metadata rather than actual pixel rotation), detects the face,
generates a 128-d face encoding, and saves everything to
models/encodings.pickle for the live recognition script to use.

Handles multi-face photos safely: if more than one face is detected in a
reference photo, that student is SKIPPED (not guessed) and flagged, so a
wrong face never silently gets encoded.

Run this once whenever you add/remove students or update their photo:
    python recognition/encoding_generator.py
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

from students.models import Student  # noqa: E402

OUTPUT_PATH = "models/encodings.pickle"


def load_image_corrected(path):
    """Load an image and apply EXIF-based orientation correction, then
    return it as an RGB numpy array (what face_recognition expects)."""
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)  # fixes phone-camera rotation
    pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def generate_encodings():
    known_encodings = []
    known_ids = []
    skipped = []

    students = Student.objects.exclude(photo="").exclude(photo__isnull=True)

    if not students.exists():
        print("No students with a photo found. Add students with photos in the admin panel first.")
        return

    print(f"Found {students.count()} student(s) with a photo. Generating encodings...\n")

    for student in students:
        photo_path = student.photo.path

        if not os.path.exists(photo_path):
            print(f"  [SKIP] {student.student_id}: photo file missing on disk ({photo_path})")
            skipped.append((student.student_id, "missing file"))
            continue

        image = load_image_corrected(photo_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"  [SKIP] {student.student_id}: no face detected in photo")
            skipped.append((student.student_id, "no face detected"))
            continue

        if len(face_locations) > 1:
            print(f"  [SKIP] {student.student_id}: {len(face_locations)} faces detected in photo "
                  f"(expected exactly 1) -> re-upload a photo with only this person's face visible")
            skipped.append((student.student_id, f"{len(face_locations)} faces detected"))
            continue

        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        known_encodings.append(encoding)
        known_ids.append(student.student_id)
        print(f"  [OK] {student.student_id} ({student.name})")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((known_encodings, known_ids), f)

    print(f"\nSaved {len(known_ids)} encoding(s) to {OUTPUT_PATH}")

    if skipped:
        print(f"\n{len(skipped)} student(s) were skipped and need attention:")
        for student_id, reason in skipped:
            print(f"  - {student_id}: {reason}")


if __name__ == "__main__":
    generate_encodings()
