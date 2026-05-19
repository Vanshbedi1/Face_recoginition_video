from deepface import DeepFace

model = DeepFace.build_model("Facenet")

def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model = model,
        enforce_detection=False
    )

    return result[0]["embedding"]
