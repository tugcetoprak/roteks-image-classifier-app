# Streamlit ONNX Image Classifier

Deploy this app to Streamlit Cloud so anyone can use your model from a browser with no installs.

## Files
- `app.py`: Streamlit app. Expects a 224x224 RGB image input; outputs one of 4 classes.
- `requirements.txt`: Python dependencies for Streamlit Cloud.
- `.streamlit/secrets.toml` (optional): Put `MODEL_URL` here if your model is hosted externally.
- `dag_model.onnx` (optional): Put your ONNX file in the repo if it's <100MB. If larger, use `MODEL_URL` instead.

## Quick Deploy (Streamlit Cloud)
1. Create a new GitHub repo and upload these files.
2. If your model is small (<100MB), add `dag_model.onnx` to the repo root. Otherwise host it (e.g., Google Drive, S3, Hugging Face) and set `MODEL_URL` in Secrets.
3. Go to https://streamlit.io → Sign in → Deploy an app → select your GitHub repo and `app.py`.
4. (Optional) In *Advanced settings → Secrets*, add:
   ```
   MODEL_URL = https://your-host/path/to/dag_model.onnx
   ```
5. Click **Deploy**. Share the app URL with others.

## Local Run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Customize
- Update `CLASS_NAMES` in `app.py` with your actual label names.
