#!/bin/bash

# Kill all background jobs when this script terminates
trap 'kill $(jobs -p)' EXIT

echo "🚀 Starting Hilltop AI (HAI)..."

# 1. Start Backend
echo "📦 Starting Backend on port 8000..."
source backend/venv/bin/activate
# Run uvicorn from the root directory context
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# 2. Start Frontend
echo "💻 Starting Frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!


echo "✅ System is running!"
echo "   - Backend: http://localhost:8000"
echo "   - Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop everything."

# Wait for both processes
wait
