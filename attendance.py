import cv2
import numpy as np
import face_recognition
import pickle
import sqlite3
from datetime import datetime, date
import pandas as pd
import os

ENC_PATH = "encodings.pkl"
DB_PATH = "attendance.db"
CSV_PATH = "attendance.csv"

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            source TEXT
        )
    ''')
    conn.commit()
    return conn

def mark_attendance_sql(conn, name, src="webcam"):
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
    if c.fetchone():
        return False
    c.execute("INSERT INTO attendance (name, date, time, source) VALUES (?, ?, ?, ?)",
              (name, today, now, src))
    conn.commit()
    return True

def append_csv(name):
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")
    row = {"name": name, "date": today, "time": now}
    df = pd.DataFrame([row])
    header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', index=False, header=header)

def main():
    if not os.path.exists(ENC_PATH):
        print("encodings.pkl not found. Run encode_faces.py first.")
        return

    with open(ENC_PATH, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

    conn = init_db()

    print("[INFO] starting webcam. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    process_every_n_frames = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not available")
            break
        frame_count += 1
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if frame_count % process_every_n_frames == 0:
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding, box in zip(encodings, boxes):
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    best_idx = np.argmin(face_distances)
                    name = known_names[best_idx]

                top, right, bottom, left = box
                top *= 2; right *= 2; bottom *= 2; left *= 2

                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if name != "Unknown":
                    added = mark_attendance_sql(conn, name)
                    if added:
                        append_csv(name)
                        print(f"[ATTENDANCE] {name} marked at {datetime.now().strftime('%H:%M:%S')}")

        cv2.imshow("Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
