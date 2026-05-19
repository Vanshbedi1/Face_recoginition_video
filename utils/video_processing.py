import cv2
from deepface import DeepFace
from utils.similarity import cosine_similarity

def process_video(video_path, target_embedding, threshold=0.7):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % int(fps) != 0:
            continue  # check 1 frame per second

        try:
            faces = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                enforce_detection=False
            )

            for face in faces:
                emb = face["embedding"]
                sim = cosine_similarity(emb, target_embedding)

                if sim > threshold:
                    time_sec = frame_count / fps
                    timestamps.append(round(time_sec, 2))

        except:
            pass

    cap.release()
    return timestamps