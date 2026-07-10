"""
Streamlit dashboard for the Face Attendance System.

Main page: browser-camera photo capture for attendance marking
(works both locally and when deployed to Streamlit Community Cloud).
Sidebar: hidden-by-default admin login (Django username OR email + password)
         -> student directory + add/delete student + attendance log viewer.

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys
import time
import pickle
import subprocess
import numpy as np
from PIL import Image
import face_recognition
import pandas as pd
import streamlit as st
import django

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from students.models import Student, AttendanceLog  # noqa: E402
from django.contrib.auth import authenticate  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.utils import timezone  # noqa: E402

ENCODINGS_PATH = "models/encodings.pickle"
DISTANCE_THRESHOLD = 0.6

st.set_page_config(page_title="Face Attendance", layout="wide", page_icon="✅")

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 900px; margin: auto; }
    h1 { text-align: center; }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 0;
    }
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 16px;
        border: 3px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helpers
# ============================================================
@st.cache_data(ttl=5)
def load_students_df():
    return pd.DataFrame(list(
        Student.objects.all().values("student_id", "name", "major", "starting_year", "total_attendance")
    ))


@st.cache_data(ttl=5)
def load_logs_df():
    logs = AttendanceLog.objects.select_related("student").values(
        "student__student_id", "student__name", "timestamp"
    ).order_by("-timestamp")
    return pd.DataFrame(list(logs))


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    with open(ENCODINGS_PATH, "rb") as f:
        return pickle.load(f)


def authenticate_flexible(identifier, password):
    user = authenticate(username=identifier, password=password)
    if user is None and "@" in identifier:
        try:
            matched = User.objects.get(email__iexact=identifier)
            user = authenticate(username=matched.username, password=password)
        except User.DoesNotExist:
            user = None
    return user


def mark_attendance(student_id):
    try:
        student = Student.objects.get(student_id=student_id)
    except Student.DoesNotExist:
        return None, "Unknown ID"

    today = timezone.now().date()
    already_marked = AttendanceLog.objects.filter(student=student, timestamp__date=today).exists()

    if already_marked:
        return student, "already marked today"

    AttendanceLog.objects.create(student=student)
    student.total_attendance += 1
    student.save()
    return student, "attendance marked!"


# ============================================================
# Session state
# ============================================================
for key, default in [("scanning", False), ("result", None), ("admin_user", None), ("show_login", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# MAIN: Take Attendance
# ============================================================
st.title("✅ Face Attendance")
st.write("")

center = st.columns([1, 2, 1])[1]

if not st.session_state.scanning and st.session_state.result is None:
    with center:
        if st.button("📷  Take Attendance", type="primary", use_container_width=True):
            st.session_state.scanning = True
            st.rerun()

elif st.session_state.result is not None:
    result = st.session_state.result

    with center:
        if result["status"] == "not_found":
            st.error("❌ Face detected, but not registered in the system.")
            if st.button("📷  Try Again", type="primary", use_container_width=True):
                st.session_state.result = None
                st.session_state.scanning = True
                st.rerun()
        else:
            if result["status"] == "attendance marked!":
                st.success(f"✅ **{result['name']}** ({result['student_id']}) — attendance marked!")
            elif result["status"] == "already marked today":
                st.info(f"ℹ️ **{result['name']}** ({result['student_id']}) — already marked today.")
            else:
                st.error("Face not recognized.")

            if st.button("📷  Scan Next Student", type="primary", use_container_width=True):
                st.session_state.result = None
                st.session_state.scanning = True
                st.cache_data.clear()
                st.rerun()

else:
    with center:
        st.info("📸 Position your face clearly, then take a photo.")
        photo = st.camera_input("Camera", label_visibility="collapsed")

        if st.button("✋ Cancel", use_container_width=True):
            st.session_state.scanning = False
            st.rerun()

        if photo is not None:
            known_encodings, known_ids = load_encodings()

            if not known_ids:
                st.error("No known faces loaded. Ask an admin to add students first.")
            else:
                image = Image.open(photo).convert("RGB")
                image_np = np.array(image)
                face_locations = face_recognition.face_locations(image_np)

                if len(face_locations) == 0:
                    st.warning("No face detected. Try again with better lighting/framing.")
                else:
                    encoding = face_recognition.face_encodings(
                        image_np, known_face_locations=[face_locations[0]]
                    )[0]
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_idx = int(distances.argmin())
                    best_distance = distances[best_idx]

                    if best_distance < DISTANCE_THRESHOLD:
                        student_id = known_ids[best_idx]
                        student, status = mark_attendance(student_id)
                        if student is None:
                            st.session_state.result = {
                                "name": None, "student_id": student_id, "status": "not_found",
                            }
                        else:
                            st.session_state.result = {
                                "name": student.name, "student_id": student.student_id, "status": status,
                            }
                    else:
                        st.session_state.result = {
                            "name": None, "student_id": None, "status": "not_found",
                        }

                    st.session_state.scanning = False
                    st.cache_data.clear()
                    st.rerun()


# ============================================================
# SIDEBAR: Admin
# ============================================================
if st.session_state.admin_user is None:
    if not st.session_state.show_login:
        if st.sidebar.button("🔒 Admin"):
            st.session_state.show_login = True
            st.rerun()
    else:
        st.sidebar.header("Admin Login")
        with st.sidebar.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Log in")

            if login_submitted:
                user = authenticate_flexible(identifier, password)
                if user is not None and user.is_staff:
                    st.session_state.admin_user = user.username
                    st.session_state.show_login = False
                    st.rerun()
                else:
                    st.error("Invalid credentials or not an admin account.")

else:
    st.sidebar.success(f"Logged in as {st.session_state.admin_user}")
    if st.sidebar.button("Log out"):
        st.session_state.admin_user = None
        st.rerun()

    tab1, tab2, tab3 = st.sidebar.tabs(["Students", "Add", "Logs"])

    with tab1:
        students_df = load_students_df()
        if not students_df.empty:
            st.dataframe(students_df[["student_id", "name", "total_attendance"]],
                         use_container_width=True, hide_index=True)

            delete_id = st.selectbox("Select student to delete", students_df["student_id"])
            if st.button("🗑️ Delete Student", type="secondary"):
                Student.objects.filter(student_id=delete_id).delete()
                subprocess.run([sys.executable, "recognition/encoding_generator.py"], capture_output=True)
                st.success(f"Deleted {delete_id} and regenerated encodings.")
                st.cache_data.clear()
                st.rerun()
        else:
            st.info("No students registered yet.")

    with tab2:
        with st.form("add_student_form", clear_on_submit=True):
            new_id = st.text_input("Student ID (e.g. S005)")
            new_name = st.text_input("Name")
            new_major = st.text_input("Major (optional)")
            new_year = st.number_input("Starting year", min_value=1990, max_value=2100, value=2024, step=1)
            new_photo = st.file_uploader("Photo (clear, front-facing)", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Add Student")

            if submitted:
                if not new_id or not new_name or not new_photo:
                    st.error("Student ID, Name, and Photo are all required.")
                elif Student.objects.filter(student_id=new_id).exists():
                    st.error(f"Student ID '{new_id}' already exists.")
                else:
                    student = Student(
                        student_id=new_id, name=new_name, major=new_major or None,
                        starting_year=int(new_year), total_attendance=0,
                    )
                    student.photo.save(new_photo.name, new_photo, save=True)
                    st.success(f"Added {new_id} - {new_name}. Regenerating encodings...")
                    result = subprocess.run(
                        [sys.executable, "recognition/encoding_generator.py"],
                        capture_output=True, text=True,
                    )
                    st.text(result.stdout[-500:] if result.stdout else "No output")
                    st.cache_data.clear()
                    st.rerun()

    with tab3:
        logs_df = load_logs_df()
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
            csv = logs_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download all logs as CSV", csv, file_name="attendance_logs.csv", mime="text/csv")
        else:
            st.info("No attendance logs yet.")
