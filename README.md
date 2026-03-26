# Face Recognition Attendance System

Simple Streamlit application for employee management and face-based attendance experiments using OpenCV and SQLite.

## Tech Stack

- Python 3.11
- Streamlit
- OpenCV
- SQLite
- NumPy
- Pandas

## Features

- Employee management page
- Face detection from camera
- Register face images from uploads or camera capture
- Local SQLite database for employee records
- Local dataset folder for face images

## Project Structure

```text
Do_an_AI/
|-- app.py
|-- core/
|   |-- face_detect.py
|   `-- save_face.py
|-- data/
|   `-- faces/
|-- db/
|   `-- database.py
|-- pages/
|   |-- employees.py
|   |-- face_detection.py
|   `-- register_face.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Data and Security Notes

- `.venv`, `attendance.db`, `__pycache__`, and generated face images are excluded from Git.
- Real face images should stay local and should not be pushed to a public repository.
- The `data/faces/.gitkeep` file is included only to preserve the folder structure.

## Current Limitation

- Face detection currently uses OpenCV Haar Cascade and is intended for learning/demo use, not production biometric security.
