import os
import pickle
import face_recognition
from pathlib import Path

DATA_DIR = Path("dataset")
OUTPUT = "encodings.pkl"

def get_image_files(folder):
    exts = (".jpg", ".jpeg", ".png")
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

def main():
    known_encodings = []
    known_names = []

    if not DATA_DIR.exists():
        print("Please create a 'dataset' folder with subfolders named by student (e.g. dataset/Alice/...)")
        return

    for person_dir in sorted(DATA_DIR.iterdir()):
        if not person_dir.is_dir(): continue
        name = person_dir.name
        print(f"[INFO] Processing {name}")
        image_files = get_image_files(person_dir)
        for img_path in image_files:
            image = face_recognition.load_image_file(str(img_path))
            boxes = face_recognition.face_locations(image, model="hog")
            encs = face_recognition.face_encodings(image, boxes)
            if len(encs) == 0:
                print(f"  [WARN] no face found in {img_path.name}, skipping")
                continue
            known_encodings.append(encs[0])
            known_names.append(name)

    print(f"[INFO] {len(known_names)} face encodings collected. Saving to {OUTPUT}")
    with open(OUTPUT, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

if __name__ == "__main__":
    main()
