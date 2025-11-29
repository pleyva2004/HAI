#!/bin/bash


# Start the FastAPI backend server
# Run this script from the HAI/ root directory
cd "$(dirname "$0")" || exit 1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload



