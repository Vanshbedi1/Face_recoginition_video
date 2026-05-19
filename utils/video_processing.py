import cv2
from utils.detection import detect_faces
from utils.embedding import get_embedding
from utils.recognition import compare_embeddings

def process_video(video_path, target_embedding):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        time_sec = frame_count / fps

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            emb = get_embedding(face)
            match, score = compare_embeddings(target_embedding, emb)

            if match:
                timestamps.append((time_sec, score))
                break

    cap.release()
    return timestamps