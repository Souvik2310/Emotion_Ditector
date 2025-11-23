# src/db.py
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        cleaned TEXT,
        predicted_label TEXT,
        predicted_idx INTEGER,
        probabilities TEXT,
        source TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(record: Dict, source: str = "single"):
    """
    record: dict with keys text, cleaned, pred_label, pred_idx, probabilities
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO predictions (text, cleaned, predicted_label, predicted_idx, probabilities, source, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("text"),
        record.get("cleaned"),
        record.get("pred_label"),
        record.get("pred_idx"),
        str(record.get("probabilities")),
        source,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def log_batch(records: List[Dict], source: str = "batch"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = []
    for r in records:
        rows.append((
            r.get("text"),
            r.get("cleaned"),
            r.get("pred_label"),
            r.get("pred_idx"),
            str(r.get("probabilities")),
            source,
            datetime.utcnow().isoformat()
        ))
    c.executemany("""
    INSERT INTO predictions (text, cleaned, predicted_label, predicted_idx, probabilities, source, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
