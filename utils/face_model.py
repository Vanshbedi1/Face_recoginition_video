from insightface.app import FaceAnalysis

# Load model once (IMPORTANT)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

def get_faces(frame):
    faces = app.get(frame)

    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.astype(int),
            "embedding": face.embedding,
            "confidence": float(face.det_score)
        })

    return results