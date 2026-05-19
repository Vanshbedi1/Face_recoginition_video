import cv2
from utils.face_model import get_faces
from utils.matching import compare_faces

def process_video(video_path, target_embedding, threshold=0.5):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Optimize: process every 5th frame
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        faces = get_faces(frame)

        for face in faces:
            sim = compare_faces(target_embedding, face["embedding"])

            if sim > threshold:
                time_sec = frame_count / fps
                timestamps.append(round(time_sec, 2))

        frame_count += 1

    cap.release()

    return timestamps