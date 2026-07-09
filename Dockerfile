FROM python:3.10-slim

# System deps needed for mysqlclient, dlib/face_recognition, and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libmariadb-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
