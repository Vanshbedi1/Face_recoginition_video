import face_recognition
import numpy as np
import os

KNOWN_FACES_DIR = "known_faces"

known_encodings = []
known_names = []

def load_known_faces():
    for file in os.listdir(KNOWN_FACES_DIR):
        image_path = os.path.join(KNOWN_FACES_DIR, file)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(file.split(".")[0])

def recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []

    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]

            name = known_names[best_match_index] if confidence > 0.5 else "Unknown"

            results.append({
                "name": name,
                "confidence": round(confidence, 2),
                "location": location
            })

    return results