import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from collections import Counter
import os

st.set_page_config(page_title="4-Image Majority Classifier", page_icon="🧮")
st.title("🧮 4-Image Majority Classifier (ONNX)")

# --- AYARLAR ---
MODEL_PATH = "dag_modelv2.onnx"
CLASS_NAMES = ["A", "BO", "E", "K"]  # <-- kendi etiketlerinle güncelle

# Modeli cache'leyerek tek sefer yükle
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model bulunamadı. `dag_model.onnx` dosyasını repo köküne ekleyin veya MODEL_URL ile indirin.")
        st.stop()
    try:
        return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error("ONNX modeli yüklenemedi.\nAyrıntı: " + str(e))
        st.stop()

ort_session = load_model()

def preprocess_pil(img_pil, size=(224, 224)):
    """PIL -> (1,3,H,W) float32 [0,1]"""
    arr = np.array(img_pil.convert("RGB"))
    arr = cv2.resize(arr, size)
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr

def predict(img_pil):
    x = preprocess_pil(img_pil)
    input_name = ort_session.get_inputs()[0].name
    preds = ort_session.run(None, {input_name: x})[0]  # (1, num_classes)
    idx = int(np.argmax(preds, axis=1)[0])
    return idx

# ---- 4 GÖRÜNTÜ YÜKLE ----
uploaded = st.file_uploader("Lütfen **tam olarak 4** görüntü yükleyin", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    if len(uploaded) != 4:
        st.warning("Tam olarak **4** görüntü yüklemelisiniz.")
        st.stop()

    images = [Image.open(f).convert("RGB") for f in uploaded]

    # Görselleri 2x2 grid olarak göster (opsiyonel)
    c1, c2 = st.columns(2)
    with c1:
        st.image(images[0], caption="Image 1", use_container_width=True)
        st.image(images[2], caption="Image 3", use_container_width=True)
    with c2:
        st.image(images[1], caption="Image 2", use_container_width=True)
        st.image(images[3], caption="Image 4", use_container_width=True)

    # Her görüntü için sınıf tahmini
    preds_idx = [predict(img) for img in images]

    # Tek tek sonuçları yaz
    st.subheader("Tekil Tahminler")
    for i, idx in enumerate(preds_idx, start=1):
        name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"Index {idx}"
        st.markdown(f"- **Image {i}:** {name}")

    # Majority vote (en çok çıkan sınıf)
    counts = Counter(preds_idx)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    winners = [cls for cls, c in most_common if c == top_count]

    if len(winners) == 1:
        final_idx = winners[0]
        final_name = CLASS_NAMES[final_idx] if 0 <= final_idx < len(CLASS_NAMES) else f"Index {final_idx}"
        st.markdown(f"### ✅ Majority vote sonucu: **{final_name}**")
    else:
        # Beraberlik durumu
        tie_names = [CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else f"Index {i}" for i in winners]
        st.warning(f"⚠️ Eşitlik: {', '.join(tie_names)}")
