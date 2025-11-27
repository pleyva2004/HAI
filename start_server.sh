#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the FastAPI server with increased timeout for long-running LLM operations
# --timeout-keep-alive: Keep connections alive for 300 seconds
# Default timeout is 60s, but we need more for question generation
python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

