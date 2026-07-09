"""
main.py

Opens the laptop webcam, detects faces in each frame, compares them against
the saved encodings (models/encodings.pickle), and on a match, marks
attendance in MariaDB via the Django ORM (once per person per day).

Run from the project root (not inside recognition/):
    python recognition/main.py

Press 'q' to quit.
"""

import os
import sys
import pickle
import django
import cv2
import face_recognition
from datetime import datetime

# --- Bootstrap Django so we can use the ORM outside manage.py ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from students.models import Student, AttendanceLog  # noqa: E402

ENCODINGS_PATH = "models/encodings.pickle"
DISTANCE_THRESHOLD = 0.6  # lower = stricter match. 0.6 is face_recognition's default.
FRAME_RESIZE_SCALE = 0.25  # shrink frame for faster detection


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"No encodings found at {ENCODINGS_PATH}. Run encoding_generator.py first.")
        sys.exit(1)
    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings, known_ids = pickle.load(f)
    print(f"Loaded {len(known_ids)} known face(s): {known_ids}")
    return known_encodings, known_ids


def mark_attendance(student_id: str) -> str:
    try:
        student = Student.objects.get(student_id=student_id)
    except Student.DoesNotExist:
        return "Unknown ID"

    today = datetime.now().date()
    already_marked = AttendanceLog.objects.filter(
        student=student, timestamp__date=today
    ).exists()

    if already_marked:
        return f"{student.name}: already marked today"

    AttendanceLog.objects.create(student=student)
    student.total_attendance += 1
    student.save()
    return f"{student.name}: attendance marked!"


def run():
    known_encodings, known_ids = load_encodings()

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open webcam. Check camera index / permissions.")
        return

    print("Webcam started. Press 'q' to quit.")
    last_status = {}

    while True:
        ok, frame = video.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = distances.argmin() if len(distances) > 0 else None

            label = "Unknown"
            if best_match_index is not None and distances[best_match_index] < DISTANCE_THRESHOLD:
                student_id = known_ids[best_match_index]
                label = student_id

                status = mark_attendance(student_id)
                if last_status.get(student_id) != status:
                    print(status)
                    last_status[student_id] = status

            scale = int(1 / FRAME_RESIZE_SCALE)
            top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
