import cv2
import numpy as np

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img / 255.0
    return face_img.flatten()[:128]
    