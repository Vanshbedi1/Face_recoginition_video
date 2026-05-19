from deepface import DeepFace

def detect_faces(frame):
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            enforce_detection=False,
            detector_backend="opencv"  # fast & stable
        )

        results = []

        for face in faces:
            region = face["facial_area"]

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]

            confidence = face.get("confidence", 0)

            results.append({
                "box": (x, y, w, h),
                "confidence": confidence
            })

        return results

    except:
        return []