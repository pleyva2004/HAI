# User Feedback Loop - How It Works

This document provides a simple overview of how the chat-based user feedback loop works in the HAI application. The goal of this feature is to allow users to ask the AI to modify or refine a question it just generated (e.g., "Make it harder", "Change the context to sports").

## 1. The User Experience (Frontend)
- **Generating the First Question:** A user types a prompt (like "algebra problem") and the AI generates a question. The system remembers the specific ID of this interaction (the "Workflow ID").
- **Depending on the Next Step:** If the user sends another text message immediately after, the chat interface assumes the user is talking about the question that was just generated.
- **Routing to Feedback:** Instead of starting a brand new task from scratch, the frontend automatically sends this message to a special "Feedback" API (`POST /api/feedback/{workflow_id}`), attaching the saved Workflow ID, along with the feedback text.

## 2. The API Layer (Backend)
- **Finding the Context:** When the backend receives the feedback, it uses the Workflow ID to find the exact state of the previous question in its memory (`completed_workflows`).
- **Updating the History:** It adds the user's new feedback to a running list called `feedback_history` and increments the `iteration_count`.
- **Resuming the Job:** The backend then restarts the AI workflow, telling it exactly where to pick back up using the same Workflow ID, passing `thread_id` to LangGraph.

## 3. The AI Workflow (LangGraph)
- **Skipping the Basics:** Because this is a revision (we already have a `feedback_history`), the workflow's smart router (`route_start`) skips the initial steps (like figuring out if the question is math or reading). It jumps straight to the "Generate" phase.
- **Process Feedback Node:** Right before asking the LLM (Claude) to generate the question, a special node (`process_feedback`) steps in. It takes the user's *original* request and tacks on *all* the new feedback the user has provided so far.
- **Generation:** Claude sees the original constraints PLUS a new section that says "The user provided the following specific requests and feedback. YOU MUST FOLLOW THESE:". It then generates the revised question.

## 4. The Final Result
- **Validation:** The new question goes through the standard validation checks.
- **Display:** The backend sends the new question back to the frontend. The chat interface replaces the old generic "Generated" badge with a new one that says **"Revised • Iteration 2"**, clearly showing the user that their feedback was applied.

---

### FAQ / Edge Cases
- **How do I stop giving feedback and start over?** If you upload a new image, or if an error occurs, the frontend clears the saved "Workflow ID" and treats your next message as a brand new request.
- **Is the feedback saved forever?** Currently, the backend stores the completed workflows in memory. If the backend server restarts, you won't be able to provide feedback on questions generated before the restart.
- **Is there a limit to how much feedback I can give?** Yes, to prevent endless loops, the system limits feedback to a maximum of 5 iterations per question (`MAX_FEEDBACK_ITERATIONS`).
