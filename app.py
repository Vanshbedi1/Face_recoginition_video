import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

from utils.detection import detect_faces
from utils.embedding import get_embedding
from utils.video_processing import process_video

st.title("Face Recognition in Video")

img_file = st.file_uploader("Upload Face Image", type=["jpg", "png"])
video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if img_file and video_file:

    # 🔹 Process image
    img = Image.open(img_file)
    frame = np.array(img)

    faces = detect_faces(frame)

    if len(faces) == 0:
        st.error("No face found in image")
    else:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        target_embedding = get_embedding(face)

        # 🔹 Save video
        os.makedirs("temp", exist_ok=True)
        video_path = "temp/input.mp4"

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # 🔹 Process video
        timestamps = process_video(video_path, target_embedding)

        # 🔹 Show result
        st.success("Face detected at:")

        for t, score in timestamps:
            mins = int(t // 60)
            secs = int(t % 60)
            st.write(f"{mins:02d}:{secs:02d} → Confidence: {round(score*100,2)}%")