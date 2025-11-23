# src/predict.py
import joblib
import numpy as np
from typing import List, Tuple, Dict
from src.preprocess import clean_text

# Adjust these paths if your models folder is elsewhere
MODEL_PATH = "models/LR_emotions.pkl"
VECT_PATH = "models/TFIDF.pkl"
LABELS_PATH = "models/Labels.pkl"

# Lazy-load and cache (simple module-level caching)
_model = None
_vectorizer = None
_labels = None
_reverse_labels = None

def load_assets():
    global _model, _vectorizer, _labels, _reverse_labels
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECT_PATH)
    if _labels is None:
        _labels = joblib.load(LABELS_PATH)
        _reverse_labels = {v: k for k, v in _labels.items()}
    return _model, _vectorizer, _labels, _reverse_labels

def predict_single(text: str) -> Dict:
    model, vectorizer, labels, reverse_labels = load_assets()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
    pred_label = reverse_labels[pred_idx]
    return {
        "text": text,
        "cleaned": cleaned,
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "probabilities": proba
    }

def predict_batch(texts: List[str]) -> List[Dict]:
    model, vectorizer, labels, reverse_labels = load_assets()
    cleaned_texts = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned_texts)
    preds = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    results = []
    for i, t in enumerate(texts):
        pidx = int(preds[i])
        results.append({
            "text": t,
            "cleaned": cleaned_texts[i],
            "pred_idx": pidx,
            "pred_label": reverse_labels[pidx],
            "probabilities": proba[i].tolist() if proba is not None else None
        })
    return results

def get_feature_importances(top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns top_n words per class (emotion) using logistic regression coef_.
    Output: {label_name: [(feature, coef), ...], ...}
    """
    model, vectorizer, labels, reverse_labels = load_assets()
    feature_names = vectorizer.get_feature_names_out()
    # For logistic regression multiclass, coef_ shape = (n_classes, n_features)
    coefs = model.coef_  # shape (n_classes, n_features)
    importances = {}
    for idx, row in enumerate(coefs):
        label = reverse_labels[idx]
        coef_pairs = list(zip(feature_names, row))
        # sort by coefficient descending to get words that positively predict class
        top = sorted(coef_pairs, key=lambda x: x[1], reverse=True)[:top_n]
        importances[label] = top
    return importances
