#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scipy",
#     "pydantic>=2.0",
#     "plotly",
#     "sqlite-vec>=0.1.0",
#     "openai>=1.0.0",     # para llamada a LLM (Gemini/OpenAI compatible)
#     "python-dotenv",
# ]
# ///

"""
sigma-zero DMAIC Agent Core
================================
Orquestador determinista para automatizar las fases Define, Measure y Analyze
de un ciclo Six Sigma DMAIC sobre datos de calidad de producción.

Ejecución:
    uv run src/dmaic_agent.py --input data/sample_defects.csv
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# ---------------------------------------------------------------------------
# 1. MODELOS DE DATOS (Pydantic) - La "Verdad de Hierro"
# ---------------------------------------------------------------------------


class DefectRecord(BaseModel):
    """Esquema inmutable para cada registro de defecto."""

    timestamp: str
    product_id: str
    defect_type: str = Field(..., description="Ej: SolderBridge, MissingComponent, Offset")
    line_id: str
    shift: str
    measured_value: float | None = None
    specification_limit_low: float | None = None
    specification_limit_high: float | None = None


class ProcessSpec(BaseModel):
    """Especificaciones del proceso cargadas desde JSON."""

    target: float
    lsl: float  # Lower Specification Limit
    usl: float  # Upper Specification Limit


# ---------------------------------------------------------------------------
# 2. FASE MEASURE - Cálculo de Capacidad de Proceso (Cpk)
# ---------------------------------------------------------------------------


def calculate_cpk(data: pd.Series, lsl: float, usl: float) -> dict:
    """Calcula Cp, Cpk y retorna un dict con el análisis."""
    mean = data.mean()
    std_dev = data.std(ddof=1)
    if std_dev == 0:
        return {
            "cp": 0,
            "cpk": 0,
            "mean": mean,
            "std_dev": std_dev,
            "status": "ERROR: No variation",
        }
    cp = (usl - lsl) / (6 * std_dev)
    cpk = min((mean - lsl) / (3 * std_dev), (usl - mean) / (3 * std_dev))
    status = "OK" if cpk >= 1.33 else ("WARNING" if cpk >= 1.0 else "CRITICAL")
    return {
        "cp": round(cp, 3),
        "cpk": round(cpk, 3),
        "mean": round(mean, 3),
        "std_dev": round(std_dev, 3),
        "status": status,
        "sigma_level": round(cpk * 3, 2),
    }


# ---------------------------------------------------------------------------
# 3. FASE ANALYZE - Búsqueda de Causa Raíz con sqlite-vec (sin alucinación)
# ---------------------------------------------------------------------------


class RootCauseEngine:
    """Busca causas raíz en una base de conocimiento vectorizada de defectos históricos."""

    def __init__(self, db_path: str = "knowledge/defect_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
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
        # Crear tabla virtual para búsqueda vectorial (sqlite-vec)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_defects USING vec0(embedding float[768])"
        )
        conn.commit()
        conn.close()

    def add_knowledge(
        self, defect_type: str, root_cause: str, solution: str, embedding: list[float]
    ):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO defects (defect_type, root_cause, solution, embedding) VALUES (?,?,?,?)",
            (defect_type, root_cause, solution, json.dumps(embedding)),
        )
        rowid = cur.lastrowid
        # También poblar la tabla de vectores
        cur.execute(
            "INSERT INTO vec_defects (rowid, embedding) VALUES (?,?)",
            (rowid, json.dumps(embedding)),
        )
        conn.commit()
        conn.close()

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.defect_type, d.root_cause, d.solution, v.distance
            FROM vec_defects v
            JOIN defects d ON v.rowid = d.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
        """,
            (json.dumps(query_embedding), top_k),
        )
        results = [
            {
                "defect_type": row[0],
                "root_cause": row[1],
                "solution": row[2],
                "similarity": round(1 - float(row[3]), 4),
            }
            for row in cur.fetchall()
        ]
        conn.close()
        return results


# ---------------------------------------------------------------------------
# 4. SÍNTESIS FINAL (Usa LLM, pero anclado a los datos)
# ---------------------------------------------------------------------------


def generate_report(
    defect_summary: dict,
    cpk_results: dict,
    root_causes: list[dict],
    pareto_chart_path: str,
) -> str:
    """Genera el reporte Markdown con la ayuda controlada del LLM (sin alucinación)."""
    # Bloque determinista del reporte
    report = f"""# Sigma-Zero DMAIC Report

## 1. 📋 Define Phase
**Top Defect:** {defect_summary["top_defect"]} ({defect_summary["count"]} \
ocurrencias, {defect_summary["percentage"]}%)

**Problem Statement:** Línea {defect_summary["primary_line"]} presenta una tasa de defectos de \
{defect_summary["dpu"]} DPU en el turno {defect_summary["primary_shift"]}, \
excediendo el límite aceptable de calidad.

## 2. 📊 Measure Phase
**Process Capability (Cpk):** {cpk_results["cpk"]} (Sigma Level: {cpk_results["sigma_level"]})
**Status:** {cpk_results["status"]}
**Mean:** {cpk_results["mean"]} | **Std Dev:** {cpk_results["std_dev"]}

El proceso {"es" if cpk_results["cpk"] >= 1.33 else "NO es"} capaz de cumplir con las \
especificaciones.

## 3. 🔍 Analyze Phase
**Root Cause Candidates (Retrieved from Knowledge Base):**
"""
    for i, rc in enumerate(root_causes, 1):
        report += (
            f"{i}. **{rc['defect_type']}**: {rc['root_cause']} "
            f"(Similitud: {rc['similarity']})\n   - Solución propuesta: {rc['solution']}\n"
        )

    report += f"\n![Pareto Chart]({pareto_chart_path})\n"
    return report


# ---------------------------------------------------------------------------
# 5. ORQUESTADOR PRINCIPAL
# ---------------------------------------------------------------------------


def run_dmaic(
    input_csv: str,
    spec_json: str = "data/process_spec.json",
    db_path: str = "knowledge/defect_memory.sqlite",
):
    # 5.1 Cargar y validar datos
    df = pd.read_csv(input_csv)
    # Validación con Pydantic (opcional pero recomendada)
    try:
        _ = [DefectRecord(**row) for row in df.to_dict(orient="records")]
    except ValidationError as e:
        print(f"❌ Error de validación de datos: {e}")
        sys.exit(1)

    # 5.2 Fase Define: Pareto de defectos
    defect_counts = df["defect_type"].value_counts()
    top_defect = defect_counts.index[0]
    top_count = int(defect_counts.iloc[0])
    total = len(df)
    pareto_data = pd.DataFrame(
        {
            "Defect": defect_counts.index,
            "Count": defect_counts.values,
            "Percentage": (defect_counts.values / total * 100).round(1),
        }
    )
    fig = px.bar(pareto_data, x="Defect", y="Count", title="Pareto de Defectos")
    pareto_path = "output/pareto_chart.html"
    fig.write_html(pareto_path)

    # 5.3 Fase Measure: Capacidad de proceso
    if "measured_value" in df.columns and df["measured_value"].notna().any():
        with Path(spec_json).open() as f:
            spec = ProcessSpec(**json.load(f))
        measurements = df["measured_value"].dropna()
        cpk_results = calculate_cpk(measurements, spec.lsl, spec.usl)
    else:
        cpk_results = {"cp": 0, "cpk": 0, "sigma_level": 0, "status": "Sin datos numéricos"}

    # 5.4 Fase Analyze: Búsqueda de causa raíz
    engine = RootCauseEngine(db_path)

    # Generar embedding simulado (en producción usarías la API de embeddings)
    def get_embedding(text: str) -> list[float]:
        # Placeholder: en producción, llama a vromlix.get_embeddings(text)
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32 - 1)
        return np.random.RandomState(seed).rand(768).tolist()

    query_emb = get_embedding(f"{top_defect} {cpk_results.get('status', '')}")
    root_causes = engine.search(query_emb)

    # 5.5 Generar reporte final
    summary = {
        "top_defect": top_defect,
        "count": top_count,
        "percentage": round(top_count / total * 100, 1),
        "dpu": round(total / df["product_id"].nunique(), 2),
        "primary_line": df["line_id"].mode()[0],
        "primary_shift": df["shift"].mode()[0],
    }
    report_md = generate_report(summary, cpk_results, root_causes, "pareto_chart.html")
    output_path = Path("output/dmaic_report.md")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(report_md)
    print(f"✅ Reporte generado: {output_path}")
    print(f"📊 Pareto chart: {pareto_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigma-Zero DMAIC Agent")
    parser.add_argument("--input", required=True, help="CSV de defectos")
    parser.add_argument("--spec", default="data/process_spec.json")
    parser.add_argument("--db", default="knowledge/defect_memory.sqlite")
    args = parser.parse_args()
    run_dmaic(args.input, args.spec, args.db)
