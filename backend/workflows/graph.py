# LangGraph workflow definition
from langgraph.graph import StateGraph, END
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

# Set entry point
workflow.set_entry_point("extract_structure")

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
agent = workflow.compile()
