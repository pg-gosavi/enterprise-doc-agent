#!/usr/bin/env bash
# Launch both FastAPI backend and Streamlit frontend in parallel.
# Usage: bash start.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Enterprise Document Intelligence Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start FastAPI backend
echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait a moment for backend to be ready
sleep 3

# Start Streamlit frontend
echo "▶ Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend  → http://localhost:8000"
echo "  API Docs → http://localhost:8000/docs"
echo "  Frontend → http://localhost:8501"
echo "  Press Ctrl+C to stop both servers."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Trap Ctrl+C and kill both
trap "echo ''; echo 'Stopping servers…'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
