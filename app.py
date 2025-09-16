import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import os
import requests

st.set_page_config(page_title="Image Classifier", page_icon="🔍")
st.title("🔍 Image Classification (ONNX)")

MODEL_PATH = "dag_model.onnx"
MODEL_URL = st.secrets.get("MODEL_URL", "")  # Optional: set in Streamlit Secrets
CLASS_NAMES = ["A", "BO", "E", "K"]  # TODO: replace with your real labels

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if MODEL_URL:
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, timeout=120)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        return MODEL_PATH
    st.error("Model not found. Put 'dag_model.onnx' in repo (if <100MB) or set MODEL_URL in secrets.")
    st.stop()

@st.cache_resource
def load_model():
    path = ensure_model()
    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error("Failed to load ONNX model. If running locally on Windows, install Microsoft VC++ 2015–2022 Redistributable (x64).\n"
                 f"Details: {e}")
        st.stop()

ort_session = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,224,224)

    input_name = ort_session.get_inputs()[0].name
    preds = ort_session.run(None, {input_name: img})[0]

    pred_idx = int(np.argmax(preds, axis=1)[0])
    result = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else f"Index {pred_idx}"

    st.markdown(f"### ✅ The image belongs to class: **{result}**")
