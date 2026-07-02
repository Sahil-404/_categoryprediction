import os
import re
import joblib
import numpy as np

#Paths
DATA_PATH  = "data/Jobs_category_data - Sheet1.csv"
MODELS_DIR = "models"

#Load models data
_model      = None
_vectorizer = None
_le         = None

def _load_models():
    global _model, _vectorizer, _le

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    vec_path   = os.path.join(MODELS_DIR, "vectorizer.pkl")
    le_path    = os.path.join(MODELS_DIR, "label_encoder.pkl")
    if not all(os.path.exists(p) for p in [model_path, vec_path, le_path]):
        raise FileNotFoundError(
            "Model files not found. Please run:\n  python ml/train_model.py"
        )

    _model      = joblib.load(model_path)
    _vectorizer = joblib.load(vec_path)
    _le         = joblib.load(le_path)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(job_title: str) -> dict:
    _load_models()
    cleaned = clean_text(job_title)
    combined = f"{cleaned} {cleaned} {cleaned}"

    vec   = _vectorizer.transform([combined])
    idx   = _model.predict(vec)[0]
    proba = _model.predict_proba(vec)[0]

    category   = _le.inverse_transform([idx])[0]
    confidence = float(np.max(proba)) * 100

    # Build all-scores dict 
    all_scores = {
        _le.inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }
    all_scores = dict(
        sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "category" : category,
        "confidence" : round(confidence, 1),
        "all_scores" : all_scores,
    }


def get_all_categories() -> list:
    """Return list of all known category labels."""
    _load_models()
    return list(_le.classes_)


