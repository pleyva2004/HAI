# All API endpoints

from fastapi import APIRouter, UploadFile, HTTPException
from backend.workflows.state import HAIState, UserInput
from backend.workflows.graph import agent
import base64

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

@router.post("/generate")
async def generate_question(request: UserInput):

     # Validate input
    if not request.image and not request.description:
        raise HTTPException(400, "Must provide either image or description")

    initial_state = HAIState.create_initial_state(request)

    try:
        # LangGraph returns dictionary that needs to be process into pydantic model
        result_unprocess = await agent.ainvoke(initial_state)
        result = HAIState.model_validate(result_unprocess)

        # Check for errors first (early failures)
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow error: {result.error}"
            )

        # Check if workflow succeeded (validation)
        if not result.validation_passed and result.validation_errors:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed after {result.generation_attempt} attempts: {result.validation_errors}"
            )


        # Check if we failed to generated question
        if not result.generated_question:
            raise HTTPException(
                status_code=500,
                detail="No question was generated"
            )


        # TODO: Store in database when implemented
        # question_id = await store_generated_question(
        #     question=result.generated_question,
        #     source_ids=[q.id for q in result.similar_questions],
        #     user_prompt=request.description,
        #     user_image=request.image
        # )


        # Return result
        return {
            "question": result.generated_question.model_dump(),
            "metadata": {
                "workflow_id": result.workflow_id,
                "number_similar_questions_used": len(result.similar_questions),
                "similar_questions_used": result.similar_questions,
                "generate_attempts": result.generation_attempt,
                "validation_passed": result.validation_passed
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        # LangSmith will capture full trace even on error
        raise HTTPException(
            status_code=500,
            detail=f"Workflow error: {str(e)}"
        )


@router.post("/upload-screenshot")
async def upload_screenshot(file: UploadFile):
    """Handle file upload and convert to base64"""

    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')

    return {"image": base64_image}
