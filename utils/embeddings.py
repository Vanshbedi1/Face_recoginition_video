from deepface import DeepFace

def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )

    return result[0]["embedding"]
