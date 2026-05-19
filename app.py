import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile

from utils.face_model import get_faces
from utils.video_processing import process_video

st.set_page_config(page_title="Face Recognition Video App")

st.title("🎯 Face Recognition in Video")

# Upload image
image_file = st.file_uploader("Upload Target Face Image", type=["jpg", "png"])

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if image_file and video_file:

    # Convert image to array
    image = Image.open(image_file)
    image = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract face embedding
    faces = get_faces(image)

    if len(faces) == 0:
        st.error("No face detected in image")
    else:
        target_embedding = faces[0]["embedding"]
        st.success("Face detected successfully!")

        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        if st.button("Start Processing"):

            with st.spinner("Processing video... ⏳"):
                timestamps = process_video(tfile.name, target_embedding)

            if len(timestamps) == 0:
                st.warning("No matching face found in video")
            else:
                st.success("Face found at these timestamps:")

                for t in timestamps:
                    st.write(f"⏱ {t} seconds")