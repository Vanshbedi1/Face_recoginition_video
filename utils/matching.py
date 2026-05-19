import numpy as np

def compare_faces(emb1, emb2):
    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )
    return similarity