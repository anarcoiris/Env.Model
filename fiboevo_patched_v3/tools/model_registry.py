import sqlite3
import json
from pathlib import Path
from typing import Optional

DB_PATH = Path("./model_registry.db")

def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_path TEXT NOT NULL,
        meta_json TEXT,
        metrics_json TEXT,
        created_at TEXT,
        code_version TEXT
    )""")
    conn.commit()
    conn.close()

def register_model(model_path: str, meta: dict = None, metrics: dict = None, code_version: str = None):
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO models (model_path, meta_json, metrics_json, created_at, code_version) VALUES (?, ?, ?, datetime('now'), ?) ",
                (model_path, json.dumps(meta) if meta else None, json.dumps(metrics) if metrics else None, code_version))
    conn.commit()
    conn.close()

def list_models():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, model_path, meta_json, metrics_json, created_at, code_version FROM models ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    models = []
    for r in rows:
        models.append({
            'id': r[0],
            'model_path': r[1],
            'meta': json.loads(r[2]) if r[2] else None,
            'metrics': json.loads(r[3]) if r[3] else None,
            'created_at': r[4],
            'code_version': r[5]
        })
    return models

def get_model(model_id: int) -> Optional[dict]:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, model_path, meta_json, metrics_json, created_at, code_version FROM models WHERE id = ?", (model_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        'id': r[0],
        'model_path': r[1],
        'meta': json.loads(r[2]) if r[2] else None,
        'metrics': json.loads(r[3]) if r[3] else None,
        'created_at': r[4],
        'code_version': r[5]
    }