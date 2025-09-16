import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import os
import requests

st.set_page_config(page_title="Image Classifier", page_icon="🔍")
st.title("🔍 Image Classification (ONNX)")

# --- Configuration ---
MODEL_PATH = "dag_model.onnx"
# You can set MODEL_URL in Streamlit Cloud -> Advanced settings -> Secrets
MODEL_URL = st.secrets.get("MODEL_URL", "")

CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]  # TODO: replace with your actual labels

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
    st.error("Model not found. Add 'dag_model.onnx' to the repo OR set MODEL_URL in secrets.")
    st.stop()

@st.cache_resource
def load_model():
    path = ensure_model()
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error("Failed to load ONNX model. If this is a local run on Windows, install the Microsoft Visual C++ 2015–2022 Redistributable (x64).\n"
                 f"Details: {e}")
        st.stop()
    return sess

ort_session = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    # Preprocess to (1,3,224,224) float32 [0,1]
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    input_name = ort_session.get_inputs()[0].name
    preds = ort_session.run(None, {input_name: img})[0]  # shape (1, num_classes)

    pred_idx = int(np.argmax(preds, axis=1)[0])
    result = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else f"Index {pred_idx}"

    st.markdown(f"### ✅ The image belongs to class: **{result}**")
