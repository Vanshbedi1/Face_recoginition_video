import cv2
from deepface import DeepFace
from utils.similarity import cosine_similarity
from utils.model import load_model

model = load_model()

def process_video(video_path, target_embedding, threshold=0.7):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    timestamps = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % int(fps) != 0:
            continue  # check 1 frame per second

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                enforce_detection=False
            )

            for face in faces:
                face_img = face["face"]

                rep = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    model=model,
                    enforce_detection=False
                )

                emb = rep[0]["embedding"]
                sim = cosine_similarity(emb, target_embedding)

                if sim > threshold:
                    time_sec = frame_count / fps
                    timestamps.add(round(time_sec, 2))

        except:
            pass

    cap.release()
    return sorted(list(timestamps))