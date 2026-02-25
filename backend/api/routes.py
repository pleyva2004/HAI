# All API endpoints

from fastapi import APIRouter, UploadFile, HTTPException
from pydantic import BaseModel
from backend.workflows.state import HAIState, UserInput, FeedbackEntry
from backend.workflows.graph import agent
from backend.workflows.nodes import process_feedback
from backend.config import MAX_FEEDBACK_ITERATIONS
from datetime import datetime, timezone
from typing import Dict
import base64

router = APIRouter()

# In-memory store for completed workflow states
completed_workflows: Dict[str, HAIState] = {}

class FeedbackRequest(BaseModel):
    feedback_text: str

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
        # LangGraph requires thread_id for checkpointing
        config = {"configurable": {"thread_id": initial_state.workflow_id}}
        result_unprocess = await agent.ainvoke(initial_state, config)
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

        # Store the completed workflow state in memory
        completed_workflows[result.workflow_id] = result

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
                "iteration_count": result.iteration_count,
                "feedback_history": [f.model_dump() for f in result.feedback_history],
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

@router.post("/feedback/{workflow_id}")
async def submit_feedback(workflow_id: str, body: FeedbackRequest):
    """
    Accept user feedback on a previously generated question.
    Re-runs the generation pipeline with feedback context.
    """
    if workflow_id not in completed_workflows:
        raise HTTPException(404, "Workflow not found or expired")

    prev_state = completed_workflows[workflow_id]

    # Check iteration limit
    if prev_state.iteration_count >= MAX_FEEDBACK_ITERATIONS:
        raise HTTPException(400, "Maximum feedback iterations reached")

    # Add feedback to history
    feedback_entry = FeedbackEntry(
        feedback_text=body.feedback_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        iteration=prev_state.iteration_count + 1
    )
    prev_state.feedback_history.append(feedback_entry)

    # Process feedback (updates description with feedback context)
    updated_state = process_feedback(prev_state)

    try:
        # Re-run the graph starting from the conditional entry point
        # We need to provide a config with thread_id since we added MemorySaver checkpointer
        config = {"configurable": {"thread_id": workflow_id}}
        result_unprocess = await agent.ainvoke(updated_state, config)
        result = HAIState.model_validate(result_unprocess)

        if not result.validation_passed and result.validation_errors:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed after {result.generation_attempt} attempts: {result.validation_errors}"
            )

        if not result.generated_question:
            raise HTTPException(
                status_code=500,
                detail="No question was generated after feedback"
            )

        # Update the stored state
        completed_workflows[workflow_id] = result

        return {
            "question": result.generated_question.model_dump(),
            "metadata": {
                "workflow_id": result.workflow_id,
                "iteration_count": result.iteration_count,
                "feedback_history": [f.model_dump() for f in result.feedback_history],
                "number_similar_questions_used": len(result.similar_questions),
                "similar_questions_used": result.similar_questions,
                "generate_attempts": result.generation_attempt,
                "validation_passed": result.validation_passed
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow error handling feedback: {str(e)}"
        )
