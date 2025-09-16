# Streamlit ONNX Image Classifier (Patched for Cloud)

This template pins packages and Python version to avoid common "installer returned a non-zero exit code" build errors on Streamlit Cloud.

## Files
- `runtime.txt` → forces Python 3.11 on Streamlit Cloud.
- `requirements.txt` → conservative pins (numpy==1.26.4, onnxruntime==1.17.1, opencv-python-headless==4.10.0.84).
- `app.py` → 224x224 RGB classifier using ONNX Runtime.
- `.streamlit/secrets.toml` (optional) → add `MODEL_URL` if model is hosted externally.

## Deploy
1) Upload these files to a **public GitHub repo**.
2) If `dag_model.onnx` < 100MB, add it to repo root. Otherwise host it elsewhere and set `MODEL_URL` in Streamlit Secrets.
3) Deploy on Streamlit Cloud targeting `app.py`.
