#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sqlite-vec>=0.1.0",
#     "numpy"
# ]
# ///

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = "knowledge/defect_memory.sqlite"


def get_embedding(text: str) -> list[float]:
    """Genera un embedding determinista (simulado) para pruebas."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32 - 1)
    return np.random.RandomState(seed).rand(768).tolist()


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY,
            defect_type TEXT,
            root_cause TEXT,
            solution TEXT,
            embedding BLOB
        )
    """)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_defects USING vec0(embedding float[768])")
    conn.commit()
    return conn


def seed_data(conn):
    historical_defects = [
        {
            "defect_type": "SolderBridge",
            "root_cause": "Exceso de pasta de soldadura en stencil debido a presión irregular "
            "de la racleta.",
            "solution": "Calibrar presión de racleta a 5kg y limpiar stencil cada 50 ciclos.",
            "text_for_embedding": "SolderBridge WARNING",
        },
        {
            "defect_type": "MissingComponent",
            "root_cause": "Falla en el alimentador (feeder) de la máquina Pick&Place por "
            "desgaste mecánico.",
            "solution": "Reemplazar engranes del feeder #4 y recalibrar pitch de avance.",
            "text_for_embedding": "MissingComponent CRITICAL",
        },
        {
            "defect_type": "Offset",
            "root_cause": "Desalineación óptica en la cámara de centrado inferior de "
            "la Pick&Place.",
            "solution": "Ejecutar rutina de calibración óptica y limpiar lente.",
            "text_for_embedding": "Offset OK",
        },
    ]

    cur = conn.cursor()
    # Limpiar tablas si existen para que sea idempotente
    cur.execute("DELETE FROM defects")
    cur.execute("DELETE FROM vec_defects")

    for d in historical_defects:
        emb = get_embedding(d["text_for_embedding"])
        cur.execute(
            "INSERT INTO defects (defect_type, root_cause, solution, embedding) VALUES (?,?,?,?)",
            (d["defect_type"], d["root_cause"], d["solution"], json.dumps(emb)),
        )
        rowid = cur.lastrowid
        cur.execute(
            "INSERT INTO vec_defects (rowid, embedding) VALUES (?,?)", (rowid, json.dumps(emb))
        )

    conn.commit()
    print(f"✅ Conocimiento histórico insertado en {DB_PATH}")


if __name__ == "__main__":
    conn = init_db()
    seed_data(conn)
    conn.close()
