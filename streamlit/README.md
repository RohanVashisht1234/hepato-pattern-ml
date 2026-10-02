# 🚀 Streamlit Deployment Bundle

This directory is an isolated, standalone deployment package for the **HepatoPattern** Streamlit clinical web application.

## 📦 Files in this Bundle
* `main.py` / `app.py`: Streamlit application source code.
* `cirrhosis.csv`: Hannover Medical School benchmark dataset (self-contained, no external network needed).
* `requirements.txt`: Python dependencies required for Streamlit Community Cloud or Docker.

## 💻 Local Execution
```bash
# From within this directory:
pip install -r requirements.txt
streamlit run main.py

# Or from project root:
streamlit run streamlit/main.py
```

## ☁️ Streamlit Community Cloud Settings
* **Repository:** `RohanVashisht1234/hepato-pattern-ml`
* **Branch:** `main`
* **Main file path:** `streamlit/main.py` (or `streamlit/app.py` or `main.py`)
