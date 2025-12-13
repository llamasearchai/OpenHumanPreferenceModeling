#!/bin/bash
# Start script for the unified FastAPI backend

cd "$(dirname "$0")"

echo "Starting OpenHumanPreferenceModeling Backend..."
echo "Backend will be available at http://localhost:8000"
echo "API docs will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
