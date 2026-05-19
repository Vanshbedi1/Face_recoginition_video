from deepface import DeepFace
from utils.model import load_model

model = load_model()

def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name="SFace",
        model = model,
        enforce_detection=False
    )

    return result[0]["embedding"]
