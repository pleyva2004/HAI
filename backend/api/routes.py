# All API endpoints

from fastapi import APIRouter

router = APIRouter()

# TODO: Add endpoints as needed
# Example endpoints from docs:
# - POST /api/generate
# - POST /api/upload-screenshot
# - POST /api/feedback
# - GET /api/questions/similar

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
