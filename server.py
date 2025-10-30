from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend import generate_answer
import json
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load flow JSON
with open("questions.json", "r") as f:
    conversation_flow = json.load(f)

# Store user sessions
user_sessions = {}


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = "default"


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_id = req.user_id
    message = req.message.strip().lower()

    # Retrieve or initialize session
    session = user_sessions.get(user_id, {"in_flow": False, "node_id": None, "context": {}, "change_target": None})
    context = session["context"]

    # Start flow trigger ---
    if not session["in_flow"] and "try your service" in message:
        session.update({"in_flow": True, "node_id": "start", "context": {}, "change_target": None})
        user_sessions[user_id] = session
        start_node = conversation_flow["start"]
        return {
            "reply": start_node["text"],
            "options": start_node.get("options", []),
            "in_flow": True,
            "node_id": "start",
            "context": {},
        }

    # Active conversation flow 
    if session["in_flow"]:
        current_id = session["node_id"]
        node = conversation_flow.get(current_id)

        if not node:
            # Safety fallback
            session["in_flow"] = False
            user_sessions[user_id] = session
            return {"reply": generate_answer(req.message), "in_flow": False}

        # Handle numbered change logic 
        # User enters the question number to change
        if current_id == "make_changes" and node.get("expect_user_input"):
            try:
                change_num = int(req.message.strip())
                summary_items = list(context.keys())
                if change_num < 1 or change_num > len(summary_items):
                    raise ValueError("Invalid number")

                target_id = summary_items[change_num - 1]
                session["change_target"] = target_id
                session["node_id"] = "await_new_answer"
                user_sessions[user_id] = session

                next_node = conversation_flow["await_new_answer"]
                target_node = conversation_flow[target_id]
                question_text = target_node["text"]

                # If the original question had options, show them again
                options = target_node.get("options", [])

                # If options exist → display them directly as choices
                if options:
                    reply_text = f"📝 Let's update your answer!\n\n{question_text}"
                    session["node_id"] = target_id  # Go back to the actual question node
                    user_sessions[user_id] = session
                    return {
                        "reply": reply_text,
                        "options": options,
                        "in_flow": True,
                        "node_id": target_id,
                        "context": context,
                    }
                else:
                    # If it was a free-text question → ask for new input
                    reply_text = f"{next_node['text']}\n\n📝 Current Question: {question_text}"
                    session["node_id"] = "await_new_answer"
                    user_sessions[user_id] = session
                    return {
                        "reply": reply_text,
                        "options": [],
                        "in_flow": True,
                        "node_id": "await_new_answer",
                        "context": context,
                    }

            except ValueError:
                return {
                    "reply": "⚠️ Please enter a valid question number (e.g., 1, 2, 3).",
                    "options": [],
                    "in_flow": True,
                    "node_id": "make_changes",
                    "context": context,
                }

        # User provides new answer for selected question
        if current_id == "await_new_answer" and node.get("expect_user_input"):
            target_id = session.get("change_target")
            if target_id and target_id in context:
                context[target_id] = req.message.strip()
                session["change_target"] = None
                session["node_id"] = "show_updated_summary"
                user_sessions[user_id] = session

                # Generate updated summary
                updated_summary = "📝 Updated Summary:\n\n"
                for idx, (key, val) in enumerate(context.items(), start=1):
                    if key in conversation_flow:
                        q_text = conversation_flow[key]["text"]
                        updated_summary += f"{idx}. {q_text} → {val}\n"

                next_node = conversation_flow["show_updated_summary"]
                return {
                    "reply": next_node["text"].replace("{{updated_summary}}", updated_summary),
                    "options": next_node.get("options", []),
                    "in_flow": True,
                    "node_id": "show_updated_summary",
                    "context": context,
                }

        # Handle normal input-based nodes 
        if node.get("expect_user_input"):
            context[node["id"]] = req.message
            session["node_id"] = node.get("next")
            user_sessions[user_id] = session
            next_node = conversation_flow.get(node.get("next"))
            if not next_node:
                session["in_flow"] = False
                return {"reply": generate_answer(req.message), "in_flow": False}
            return {
                "reply": next_node["text"],
                "options": next_node.get("options", []),
                "in_flow": True,
                "node_id": node["next"],
                "context": context,
            }

        # Handle option-based nodes 
        if "options" in node and node["options"]:
            # Convert all option labels to lowercase for comparison
            option_labels = [opt["label"].lower() for opt in node["options"]]
            
            # Exact match (case-insensitive)
            selected = next((opt for opt in node["options"] if opt["label"].lower() == message), None)

            # Try fuzzy match if no exact match
            if not selected:
                close_matches = difflib.get_close_matches(message, option_labels, n=1, cutoff=0.7)
                if close_matches:
                    # Auto-correct with the closest match
                    corrected_label = close_matches[0]
                    selected = next((opt for opt in node["options"] if opt["label"].lower() == corrected_label), None)

            # If still no match → ask user if they want to restart
            if not selected:
                session["node_id"] = "mistake_prompt"
                session["in_flow"] = True
                user_sessions[user_id] = session
                next_node = conversation_flow["mistake_prompt"]
                return {
                    "reply": "😕 Oops — it looks like your reply didn't match the available options. Would you like to start the requirement flow again?",
                    "options": next_node.get("options", []),
                    "in_flow": True,
                    "node_id": "mistake_prompt",
                    "context": context,
                }

            # Valid (or corrected) selection
            context[node["id"]] = selected["label"]
            next_id = selected["next"]

        else:
            next_id = node.get("next")

        # Transition to next node 
        if next_id:
            next_node = conversation_flow.get(next_id)
            if not next_node:
                session["in_flow"] = False
                user_sessions[user_id] = session
                return {"reply": generate_answer(req.message), "in_flow": False}

            session["node_id"] = next_id
            user_sessions[user_id] = session

            # Generate numbered summary 
            if next_id == "show_summary":
                summary_text = "📝 Summary of your responses:\n\n"

                for idx, (key, val) in enumerate(context.items(), start=1):
                    node = conversation_flow.get(key, {})
                    # Only include real user questions (not flow/system nodes)
                    if node and "options" in node and not key.startswith("show_") and not key.startswith("submit_"):
                        q_text = node.get("text", "").strip()
                        if q_text:
                            summary_text += f"{idx}. {q_text} → {val}\n"

                return {
                    "reply": next_node["text"].replace("{{summary}}", summary_text),
                    "options": next_node.get("options", []),
                    "in_flow": True,
                    "node_id": "show_summary",
                    "context": context,
                }

            # End of flow 
            if next_id == "submit_response":
                session["in_flow"] = False
                user_sessions[user_id] = session
                return {
                    "reply": next_node["text"],
                    "options": [],
                    "in_flow": False,
                    "node_id": "submit_response",
                    "context": context,
                }

            # Normal flow step 
            return {
                "reply": next_node["text"],
                "options": next_node.get("options", []),
                "in_flow": True,
                "node_id": next_id,
                "context": context,
            }

        # No next node
        session["in_flow"] = False
        user_sessions[user_id] = session
        # return {"reply": "✅ Thank you! Your responses have been recorded.", "in_flow": False}

    # Normal chat mode 
    return {"reply": generate_answer(req.message), "in_flow": False}


@app.get("/")
def root():
    return {"message": "✅ Audio Chatbot API is running!"}
