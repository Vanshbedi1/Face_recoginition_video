import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.face_utils import load_known_faces, recognize_faces

st.title("Face Recognition System")

load_known_faces()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    frame = np.array(image)

    results = recognize_faces(frame)

    for res in results:
        top, right, bottom, left = res["location"]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        label = f"{res['name']} ({res['confidence']})"

        cv2.putText(frame, label, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    st.image(frame, channels="BGR")