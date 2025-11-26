#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the FastAPI server
python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

