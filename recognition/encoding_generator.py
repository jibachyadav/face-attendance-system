"""
encoding_generator.py

Reads every Student record that has a photo uploaded (via Django admin),
detects the face in that photo, generates a 128-d face encoding, and saves
everything to models/encodings.pickle for the live recognition script to use.

Run this once whenever you add/remove students or update their photo:
    python recognition/encoding_generator.py
"""

import os
import sys
import pickle
import django
import face_recognition

# --- Bootstrap Django so we can use the ORM outside manage.py ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from students.models import Student  # noqa: E402

OUTPUT_PATH = "models/encodings.pickle"


def generate_encodings():
    known_encodings = []
    known_ids = []

    students = Student.objects.exclude(photo="").exclude(photo__isnull=True)

    if not students.exists():
        print("No students with a photo found. Add students with photos in the admin panel first.")
        return

    print(f"Found {students.count()} student(s) with a photo. Generating encodings...")

    for student in students:
        photo_path = student.photo.path

        if not os.path.exists(photo_path):
            print(f"  [SKIP] {student.student_id}: photo file missing on disk ({photo_path})")
            continue

        image = face_recognition.load_image_file(photo_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"  [SKIP] {student.student_id}: no face detected in photo")
            continue
        if len(face_locations) > 1:
            print(f"  [WARN] {student.student_id}: {len(face_locations)} faces found, using the first one")

        encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
        known_encodings.append(encoding)
        known_ids.append(student.student_id)
        print(f"  [OK] {student.student_id} ({student.name})")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((known_encodings, known_ids), f)

    print(f"\nSaved {len(known_ids)} encoding(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_encodings()
