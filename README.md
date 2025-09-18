# 🎓 Automated Student Attendance Tracking System

A Python-based project that uses **face recognition** to mark student attendance automatically via webcam.

## 🚀 Features
- Real-time face recognition
- Attendance stored in **CSV** and **SQLite**
- Prevents duplicate entries for the same student in a day
- Easily extendable

## 🛠️ Tech
- Python 3.x
- OpenCV
- face_recognition (dlib)
- SQLite + Pandas

## ⚡ Usage
1. Add student images inside `dataset/Name/` (e.g., `dataset/Alice/alice1.jpg`).
2. Generate encodings:
   ```bash
   python encode_faces.py
