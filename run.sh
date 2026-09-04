#!/usr/bin/env bash
# ChargeShield AI — Runner Script
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Locate virtual environment python/streamlit
if [ -d "$DIR/.venv" ]; then
    PYTHON="$DIR/.venv/bin/python"
    STREAMLIT="$DIR/.venv/bin/streamlit"
    UVICORN="$DIR/.venv/bin/uvicorn"
    PYTEST="$DIR/.venv/bin/pytest"
else
    PYTHON="python3"
    STREAMLIT="streamlit"
    UVICORN="uvicorn"
    PYTEST="pytest"
fi

COMMAND="${1:-dashboard}"

case "$COMMAND" in
    dashboard)
        echo "🛡️  Starting ChargeShield AI Streamlit Dashboard..."
        "$STREAMLIT" run dashboard/app.py --server.port 8501
        ;;
    api)
        echo "🚀 Starting ChargeShield AI FastAPI Backend..."
        "$UVICORN" chargeshield.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    all)
        echo "🛡️  Starting ChargeShield AI (FastAPI + Streamlit)..."
        "$UVICORN" chargeshield.api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        trap "kill $API_PID 2>/dev/null || true" EXIT
        "$STREAMLIT" run dashboard/app.py --server.port 8501
        ;;
    test)
        echo "🧪 Running Pytest test suite..."
        "$PYTEST" -v tests/
        ;;
    generate-data)
        echo "📊 Generating synthetic dataset..."
        "$PYTHON" scripts/generate_data.py --num-txns "${2:-30000}"
        ;;
    train)
        echo "🧠 Training XGBoost + Isolation Forest models..."
        "$PYTHON" scripts/train.py
        ;;
    evaluate)
        echo "📈 Evaluating model on held-out test data..."
        "$PYTHON" scripts/evaluate.py
        ;;
    *)
        echo "Usage: ./run.sh [dashboard|api|all|test|train|evaluate|generate-data]"
        exit 1
        ;;
esac
