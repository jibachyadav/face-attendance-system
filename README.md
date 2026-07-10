# face-attendance-system

A real-time face recognition attendance system built with Django, MariaDB, OpenCV, and `face_recognition`, with a Streamlit dashboard for taking attendance and managing students.

## Features

- **Face detection & recognition** using the `face_recognition` library (dlib HOG detector + 128-d face embeddings)
- **Django + MariaDB backend** for student records and attendance logs
- **Django admin panel** for managing students and viewing logs
- **Streamlit dashboard** with:
  - Fully automatic, continuous camera scanning for attendance (no manual capture needed)
  - Once-per-day attendance limiting per student
  - Admin login (real Django accounts) to add/delete students and view attendance history
  - CSV export of attendance logs
- **Docker Compose** setup for Django + MariaDB
- **GitHub Actions CI** (lint + Django checks on every push)
- **Basic accuracy/threshold evaluation script** for documenting recognition behavior

## Tech Stack

- **Backend**: Django 5.2, MariaDB
- **Face Recognition**: `face_recognition` (dlib), OpenCV
- **Dashboard**: Streamlit, Pandas, Plotly
- **DevOps**: Docker, Docker Compose, GitHub Actions

## Architecture

Webcam / Camera
│
▼
Face Detection (HOG) ──▶ 128-d Encoding ──▶ Compare vs known encodings
│
match found?  ──┴── no match
│                │
▼                ▼
Check "already marked today"   "Not found"
│
Django ORM write
│
▼
MariaDB (Student, AttendanceLog)



## Project Structure

face-attendance-system/
├── config/                  # Django project settings
├── students/                 # Django app: Student & AttendanceLog models, admin
│   └── management/commands/  # reset_attendance management command
├── recognition/
│   ├── encoding_generator.py # Builds face encodings from student photos in the DB
│   ├── main.py                # Standalone OpenCV webcam attendance script
│   └── evaluate.py            # Accuracy/threshold evaluation script
├── dashboard/
│   └── app.py                 # Streamlit dashboard (attendance + admin panel)
├── test_images/                # Labeled test images for evaluate.py
├── models/
│   └── encodings.pickle       # Generated face encodings (not committed)
├── docker-compose.yml
├── Dockerfile
├── packages.txt                # System deps (for cloud deployment, if used)
├── requirements.txt
└── manage.py



## Setup

### 1. Environment

```bash
conda create -n face_attendance python=3.10 -y
conda activate face_attendance
pip install -r requirements.txt
```

`face_recognition` depends on `dlib`, which needs a C++ compiler and CMake:
```bash
sudo apt install -y cmake build-essential pkg-config libmariadb-dev
```

### 2. Database (MariaDB)

```bash
sudo apt install -y mariadb-server
sudo mysql -u root
```
```sql
CREATE DATABASE attendance_db;
CREATE USER 'admin'@'localhost' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON attendance_db.* TO 'admin'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

### 4. Migrate & create an admin account

```bash
python manage.py migrate
python manage.py createsuperuser
```

## Usage

### Add students

Go to `http://127.0.0.1:8000/admin/` (run `python manage.py runserver` first), log in, and add students with a photo under **Students**.

### Generate face encodings

Run this whenever students are added, removed, or their photo changes:
```bash
python recognition/encoding_generator.py
```

### Take attendance

**Option A — Streamlit dashboard (recommended):**
```bash
streamlit run dashboard/app.py
```
Click "Take Attendance" — the camera scans continuously and automatically marks attendance when a known face is recognized, looping for the next person. Admin login (sidebar) allows adding/deleting students and viewing logs.

**Option B — standalone script:**
```bash
python recognition/main.py
```

### Reset attendance (for testing/demo)

```bash
python manage.py reset_attendance            # delete all logs
python manage.py reset_attendance --today     # delete only today's logs
python manage.py reset_attendance --full      # also reset total_attendance counters
```

### Run with Docker

```bash
docker compose up -d
docker compose exec web python manage.py migrate
docker compose exec web python manage.py createsuperuser
```
Note: the Docker setup runs Django + MariaDB only. The webcam recognition scripts must be run on the host machine, since containers don't have access to physical camera hardware by default.

### Evaluate recognition accuracy

Add labeled test images to `test_images/` (`<student_id>_description.jpg`, or `unknown_description.jpg` for people not in the system), then:
```bash
python recognition/evaluate.py
```

## Evaluation Results (baseline)

A baseline run using the 5 registered students' reference photos plus one unrelated "unknown" photo:

| Metric | Result |
|---|---|
| Correct matches | 5/5 |
| Correct unknown rejections | 1/1 |
| False accepts | 0 |
| False rejects | 0 |
| Distance threshold | 0.6 |

**Important caveat:** this baseline reuses the exact reference photos used to generate the encodings, so the 5/5 match result is expected (self-matching a photo against its own encoding always succeeds) rather than a measure of real-world generalization. The more meaningful data point is the unknown-person rejection at distance `0.662`, just above the `0.6` threshold. A more rigorous evaluation would use separate, different photos of each person (different angle/lighting/day) — this is a documented limitation, not a claimed accuracy metric.

## Known Limitations

- **No model training**: this project uses a pretrained face detector and encoder (`face_recognition`/dlib). Recognition is based on distance-threshold matching, not a trained classifier — there are no accuracy/loss curves in the ML-training sense.
- **EXIF orientation**: phone camera photos often store rotation as metadata rather than rotating actual pixels. The encoding generator corrects for this (`ImageOps.exif_transpose`), but this was a real bug encountered during development, documented here for transparency.
- **Single face per frame assumption**: the live recognition scripts process the first detected face per frame; multiple simultaneous faces are only handled by the encoding generator's multi-face-photo safety check, not by the live attendance loop.
- **Local-only continuous scanning**: the Streamlit dashboard's continuous auto-scan uses `cv2.VideoCapture`, which accesses the camera of the machine running the script. This works when run locally but cannot be deployed to a remote host (e.g. Streamlit Community Cloud) without switching to browser-based camera capture (`st.camera_input`), which requires a manual click per photo instead of continuous auto-detection.
- **Threshold tuning, not trained calibration**: the `0.6` distance threshold is the `face_recognition` library's default, not a value tuned against a proper validation set for this specific population of faces.



