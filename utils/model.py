import streamlit as st
from deepface import DeepFace

@st.cache_resource
def load_model():
    return DeepFace.build_model("Facenet")