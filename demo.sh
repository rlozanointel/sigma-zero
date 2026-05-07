#!/usr/bin/env bash

set -e

# Change to the directory of the script
cd "$(dirname "$0")"

echo "=========================================="
echo "🚀 INICIANDO SIGMA-ZERO DEMO 🚀"
echo "=========================================="

echo "[1/3] Poblando base de conocimiento histórica (SQLite-vec)..."
uv run src/seed_knowledge.py

echo ""
echo "[2/3] Ejecutando Orquestador DMAIC Agent..."
uv run main.py --input data/sample_defects.csv --spec data/process_spec.json --db knowledge/defect_memory.sqlite

echo ""
echo "[3/3] Demo Finalizado."
echo "➡️  Revisa la carpeta 'output/' para ver tu reporte Markdown y el gráfico de Pareto interactivo."
echo "=========================================="
