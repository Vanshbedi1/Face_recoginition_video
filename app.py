import streamlit as st
import tempfile
from utils.embedding import get_embedding
from utils.video_processing import process_video

st.title("Face Recognition in Video 🎥")

# Upload image
image_file = st.file_uploader("Upload Face Image", type=["jpg", "png"])

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if image_file and video_file:
    with tempfile.NamedTemporaryFile(delete=False) as img_tmp:
        img_tmp.write(image_file.read())
        img_path = img_tmp.name

    with tempfile.NamedTemporaryFile(delete=False) as vid_tmp:
        vid_tmp.write(video_file.read())
        vid_path = vid_tmp.name

    st.info("Processing... ⏳")

    # Step 1: embedding
    with st.spinner("Loading AI model... first run may take time ⏳"):
        target_embedding = get_embedding(img_path)

    # Step 2: process video
    timestamps = process_video(vid_path, target_embedding)

    if timestamps:
        st.success(f"Face found at timestamps (seconds): {timestamps}")
    else:
        st.warning("Face not found in video")