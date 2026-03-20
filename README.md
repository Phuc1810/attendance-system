# Project: Face Recognition Attendance System

# Technology:
- Python, OpenCV, Streamlit, SQLite
# How to run:
- activate venv and run streamlit
# Environment:
- Window 10, Python 3.11, VS Code
# Library: 
- Streamlit and opencv-python
# Result:
- Streamlit can run, Camera laptop and Camera C200 (external devices) are activate

# Project structure:
Do_an_AI/
├── app.py
├── core/
├── data/
├── db/
├── pages/
├── attendance.db
├── README.md
└── .gitignore

# Week 1 Progress (Completed)
- Installed Python 3.11
- Configured virtual environment (`.venv`)
- Installed required libraries
- Ran Streamlit succcessfully
- Tested both cameras sucessfully

# Week 2 Process (Completed)
- Created project structure
- Built streamlit multi-page foundation
- Created SQLite database
- Implemented employee management page
- Added employee data display with pandas

# Week 3 Process (Completed)
- Create core/face_detect.py: this is the AI part that processes facial recognition in the camera
- Loaded Haar Cascade face detection model from OpenCV
- Implenment detect_faces(frame) in face_detect.py: When a face appears in a camera's frame, system will create a frame to
capture the face
- Detected faces from both cameras successfully
- Displayed face detection results on Streamlit page 