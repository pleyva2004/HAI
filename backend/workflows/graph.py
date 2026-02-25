# LangGraph workflow definition
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from backend.workflows.state import HAIState
from backend.workflows.nodes import extract_structure, classify_question, retrieve_examples, generate_question, validate_output, should_validate, validation_decision

# Initialize Graph
workflow = StateGraph(HAIState)

# Add nodes
workflow.add_node("extract_structure", extract_structure)
workflow.add_node("classify_question", classify_question)
workflow.add_node("retrieve_examples", retrieve_examples)
workflow.add_node("generate_question", generate_question)
workflow.add_node("validate_output", validate_output)

def route_start(state: HAIState) -> str:
    # If we have feedback_history, skip extraction and go straight to generation
    if state.feedback_history and len(state.feedback_history) > 0:
        return "generate_question"
    return "extract_structure"

# Set entry point
workflow.set_conditional_entry_point(
    route_start,
    {
        "extract_structure": "extract_structure",
        "generate_question": "generate_question"
    }
)

# Add sequential edges
workflow.add_edge("extract_structure", "classify_question")
workflow.add_edge("classify_question", "retrieve_examples")
workflow.add_edge("retrieve_examples", "generate_question")

# Add conditional edges
workflow.add_conditional_edges(
    "generate_question",
    should_validate,
    {
        "validate": "validate_output",
        "end": END
    }
)


workflow.add_conditional_edges(
    "validate_output",
    validation_decision,
    {
        "success": END,
        "regenerate": "generate_question",
        "failed": END
    }
)

# Compile
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
