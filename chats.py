import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
from db import supabase, insert_row, delete_rows, update_row
from models import ChatRequest, CreateChatRequest
from llm_manager import generate_answer_async
from flows import flow_manager, conversation_flow
from greetings import is_greeting, GREETING_REPLY
from utils import normalize_text
from config import MAX_CONTEXT_TOKENS
from faiss_manager import add as faiss_add

router = APIRouter(prefix="/chats")
logger = logging.getLogger("swarai.chats")

# in-memory session store (keeps same behavior as original server.py)
user_sessions: Dict[str, Dict[str, Any]] = {}

# model context tracking (tokens)
model_context: Dict[str, Dict[str, int]] = {}

def reset_model_context(session_id: str):
    model_context[session_id] = {"tokens_used": 0}

def update_model_context(session_id: str, tokens: int):
    if session_id not in model_context:
        reset_model_context(session_id)
    model_context[session_id]["tokens_used"] += tokens

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))

@router.get("/get_chats/{user_id}")
async def get_chats(user_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        r = supabase.table('chat_sessions').select('*').eq('user_id', user_id).order('updated_at', desc=True).execute()
        return r.data or []
    except Exception:
        logger.exception('get_chats failed')
        raise HTTPException(status_code=500, detail='Failed to fetch chats')

@router.get("/get_chat/{chat_id}")
async def get_chat(chat_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        r = supabase.table('chat_messages').select('*').eq('session_id', chat_id).order('created_at', desc=False).execute()
        return r.data or []
    except Exception:
        logger.exception('get_chat failed')
        raise HTTPException(status_code=500, detail='Failed to fetch chat messages')

@router.post("/create_chat")
async def create_chat(payload: CreateChatRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        new_chat_id = str(uuid4())
        now = datetime.utcnow().isoformat()
        insert_row('chat_sessions', {
            'id': new_chat_id,
            'user_id': payload.user_id,
            'title': payload.title,
            'created_at': now,
            'updated_at': now
        })
        # attach to in-memory session
        session = user_sessions.setdefault(payload.user_id, {})
        session['session_id'] = new_chat_id
        user_sessions[payload.user_id] = session
        return {'id': new_chat_id, 'title': payload.title, 'user_id': payload.user_id}
    except Exception:
        logger.exception('create_chat failed')
        raise HTTPException(status_code=500, detail='Failed to create chat')

@router.delete("/delete_chat/{chat_id}")
async def delete_chat(chat_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        delete_rows('chat_messages', {'session_id': chat_id})
        delete_rows('chat_sessions', {'id': chat_id})
        return {'success': True}
    except Exception:
        logger.exception('delete_chat failed')
        raise HTTPException(status_code=500, detail='Failed to delete chat')

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_id = req.user_id or "guest"
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail='Empty message')
    
    # Guest / temporary session check
    is_guest = user_id.startswith("guest_") or user_id == "guest"
    if is_guest:
        # Handle guest chat completely in memory — not persisted in Supabase
        guest_session = user_sessions.get(user_id, {"messages": []})
        guest_session["messages"].append({"role": "user", "content": message})

        # Generate simple or model-based reply (reuse your existing async generator)
        async def _guest_reply(text: str) -> str:
            parts = await generate_answer_async(text)
            if isinstance(parts, list) and parts:
                return parts[0]
            return parts if isinstance(parts, str) else str(parts)

        reply_text = await _guest_reply(message)
        guest_session["messages"].append({"role": "assistant", "content": reply_text})
        user_sessions[user_id] = guest_session

        return {
            "reply": reply_text,
            "guest": True,
            "in_flow": False,
            "session_id": user_id,   
            "messages": guest_session["messages"],
            "message": "⚡ Temporary guest chat (not saved)"
        }

    # prepare in-memory session defaults (preserve original behavior)
    default_session = {
        "in_flow": False,
        "node_id": None,
        "context": {},
        "change_target": None,
        "authenticated": False,
        "session_id": None,
    }
    session = user_sessions.get(user_id, {})
    for k, v in default_session.items():
        session.setdefault(k, v)
    user_sessions[user_id] = session
    is_demo_user = session.get("authenticated", False)

    async def _get_reply_text(text: str) -> str:
        parts = await generate_answer_async(text)
        if isinstance(parts, list) and parts:
            return parts[0]
        return parts if isinstance(parts, str) else str(parts)

    session_id = session.get("session_id")

    # 1) Flow trigger: if not in flow and message triggers service flow (use flow trigger detection from backend.py)
    # backend.py defines matches_services_trigger -- import robustly
    try:
        from backend import matches_services_trigger
    except Exception:
        try:
            from .backend import matches_services_trigger
        except Exception:
            matches_services_trigger = lambda t: False

    if not session["in_flow"] and matches_services_trigger(message):
        session.update({"in_flow": True, "node_id": "start", "context": {}, "change_target": None})
        user_sessions[user_id] = session
        start_node = conversation_flow.get("start", {"text": "Starting..."})
        reply_text = start_node.get("text", "Starting...")
        options = start_node.get("options", [])
        return {
            "reply": reply_text,
            "options": options,
            "in_flow": True,
            "node_id": "start",
            "context": {},
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply_text}
            ],
        }

    # 2) If in flow -> delegate to FlowManager logic (preserves your server.py flow)
    if session["in_flow"]:
        current_id = session["node_id"]
        node = conversation_flow.get(current_id)

        if not node:
            session["in_flow"] = False
            user_sessions[user_id] = session
            reply_text = await _get_reply_text(req.message)
            return {"reply": reply_text, "in_flow": False}

        # handle make_changes expecting numeric input
        if current_id == "make_changes" and node.get("expect_user_input"):
            try:
                change_num = int(message.strip())
                summary_items = list(session["context"].keys())
                if change_num < 1 or change_num > len(summary_items):
                    raise ValueError("Invalid number")
                target_id = summary_items[change_num - 1]
                session["change_target"] = target_id
                session["node_id"] = "await_new_answer"
                user_sessions[user_id] = session
                next_node = conversation_flow.get("await_new_answer", {})
                target_node = conversation_flow.get(target_id, {})
                question_text = target_node.get("text", "")
                options = target_node.get("options", [])
                if options:
                    reply_text = f"📝 Let's update your answer!\n\n{question_text}"
                    session["node_id"] = target_id
                    user_sessions[user_id] = session
                    return {"reply": reply_text, "options": options, "in_flow": True, "node_id": target_id, "context": session["context"]}
                else:
                    reply_text = f"{next_node.get('text', '')}\n\n📝 Current Question: {question_text}"
                    session["node_id"] = "await_new_answer"
                    user_sessions[user_id] = session
                    return {"reply": reply_text, "options": [], "in_flow": True, "node_id": "await_new_answer", "context": session["context"]}
            except Exception:
                return {"reply": "⚠️ Please enter a valid question number (e.g., 1, 2, 3).", "options": [], "in_flow": True, "node_id": "make_changes", "context": session["context"]}

        # await_new_answer branch
        if current_id == "await_new_answer" and node.get("expect_user_input"):
            target_id = session.get("change_target")
            if target_id and target_id in session["context"]:
                session["context"][target_id] = message.strip()
                session["change_target"] = None
                session["node_id"] = "show_updated_summary"
                user_sessions[user_id] = session
                updated_summary = "📝 Updated Summary:\n\n"
                for idx, (key, val) in enumerate(session["context"].items(), start=1):
                    if key in conversation_flow:
                        q_text = conversation_flow[key].get("text", "")
                        updated_summary += f"{idx}. {q_text} → {val}\n"
                next_node = conversation_flow.get("show_updated_summary", {})
                return {
                    "reply": next_node.get("text", "").replace("{{updated_summary}}", updated_summary),
                    "options": next_node.get("options", []),
                    "in_flow": True,
                    "node_id": "show_updated_summary",
                    "context": session["context"],
                }

        # general expect_user_input handling
        if node.get("expect_user_input"):
            # store raw input into context (preserve original semantics)
            session["context"][node.get("id")] = message
            session["node_id"] = node.get("next")
            user_sessions[user_id] = session
            next_node = conversation_flow.get(node.get("next"))
            if not next_node:
                session["in_flow"] = False
                user_sessions[user_id] = session
                reply_text = await _get_reply_text(req.message)
                return {"reply": reply_text, "in_flow": False}
            return {"reply": next_node.get("text", ""), "options": next_node.get("options", []), "in_flow": True, "node_id": node.get("next"), "context": session["context"]}

        # options handling — use flow_manager.find_best_option for fuzzy match
        if "options" in node and node["options"]:
            selected = flow_manager.find_best_option(node["options"], message)
            if not selected:
                session["node_id"] = "mistake_prompt"
                session["in_flow"] = True
                user_sessions[user_id] = session
                next_node = conversation_flow.get("mistake_prompt", {})
                return {"reply": "😕 Oops — that didn’t match any option. Restart requirement flow?", "options": next_node.get("options", []), "in_flow": True, "node_id": "mistake_prompt", "context": session["context"]}
            session["context"][node["id"]] = selected["label"]
            next_id = selected.get("next")
        else:
            next_id = node.get("next")

        if next_id:
            next_node = conversation_flow.get(next_id)
            if not next_node:
                session["in_flow"] = False
                user_sessions[user_id] = session
                reply_text = await _get_reply_text(req.message)
                return {"reply": reply_text, "in_flow": False}
            session["node_id"] = next_id
            user_sessions[user_id] = session

            # show summary logic (preserve original)
            if next_id == "show_summary":
                summary_text = "📝 Summary of your responses:\n\n"
                for idx, (key, val) in enumerate(session["context"].items(), start=1):
                    node_local = conversation_flow.get(key, {})
                    if node_local and "options" in node_local and not key.startswith(("show_", "submit_")):
                        q_text = node_local.get("text", "").strip()
                        if q_text:
                            summary_text += f"{idx}. {q_text} → {val}\n"
                next_node = conversation_flow.get("show_summary", {})
                return {"reply": next_node.get("text", "").replace("{{summary}}", summary_text), "options": next_node.get("options", []), "in_flow": True, "node_id": "show_summary", "context": session["context"]}

            if next_id == "submit_response":
                session["in_flow"] = False
                user_sessions[user_id] = session
                return {"reply": next_node.get("text", ""), "options": [], "in_flow": False, "node_id": "submit_response", "context": session["context"]}

            return {"reply": next_node.get("text", ""), "options": next_node.get("options", []), "in_flow": True, "node_id": next_id, "context": session["context"]}

        session["in_flow"] = False
        user_sessions[user_id] = session

    # 3) Greeting handling (robust)
    if is_greeting(message):
        reply_text = GREETING_REPLY
        try:
            # persist to supabase if configured
            session_id = req.session_id or session.get("session_id")
            if not session_id and supabase:
                existing = supabase.table("chat_sessions").select("id", "title").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
                if existing.data:
                    session_id = existing.data[0]["id"]
                else:
                    session_id = str(uuid4())
                    now = datetime.utcnow().isoformat()
                    insert_row('chat_sessions', {
                        "id": session_id, "user_id": user_id, "title": "New Chat", "created_at": now, "updated_at": now
                    })
            session["session_id"] = session_id
            user_sessions[user_id] = session

            now = datetime.utcnow().isoformat()
            if supabase:
                insert_row('chat_messages', {
                    "session_id": session_id, "role": "user", "content": message, "created_at": now
                })
                insert_row('chat_messages', {
                    "session_id": session_id, "role": "assistant", "content": reply_text, "created_at": now
                })
                # set a smart title if still default
                session_row = supabase.table("chat_sessions").select("title").eq("id", session_id).execute()
                current_title = (session_row.data[0]["title"] if session_row.data else "New Chat")
                if current_title == "New Chat" and message.strip():
                    smart_title = message[:50].strip().capitalize()
                    update_row('chat_sessions', {'id': session_id}, {'title': smart_title, 'updated_at': now})
        except Exception:
            logger.exception("Supabase save failed (greeting).")
        return {
            "reply": reply_text,
            "in_flow": False,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply_text}
            ],
        }

    # 4) Standard generation
    reply_text = await _get_reply_text(message)

    # token and context management (if wanted)
    try:
        session_id = req.session_id or session.get("session_id")
        if session_id not in model_context:
            reset_model_context(session_id)
        tokens_this_msg = estimate_tokens(message) + estimate_tokens(reply_text)
        update_model_context(session_id, tokens_this_msg)
        if model_context[session_id]["tokens_used"] > MAX_CONTEXT_TOKENS - 100:
            reset_model_context(session_id)

        # ensure session exists and persist messages
        if not session_id and supabase:
            existing = supabase.table("chat_sessions").select("id", "title").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
            if existing.data:
                session_id = existing.data[0]['id']
            else:
                session_id = str(uuid4())
                now = datetime.utcnow().isoformat()
                insert_row('chat_sessions', {
                    "id": session_id, "user_id": user_id, "title": "New Chat", "created_at": now, "updated_at": now
                })

        session["session_id"] = session_id
        user_sessions[user_id] = session

        now = datetime.utcnow().isoformat()
        if supabase:
            insert_row('chat_messages', {
                "session_id": session_id, "role": "user", "content": message, "created_at": now
            })
            insert_row('chat_messages', {
                "session_id": session_id, "role": "assistant", "content": reply_text, "created_at": now
            })
            # update smart title if new
            session_row = supabase.table("chat_sessions").select("title").eq("id", session_id).execute()
            current_title = (session_row.data[0]["title"] if session_row.data else "New Chat")
            if current_title == "New Chat" and message.strip():
                smart_title = message[:50].strip().capitalize()
                update_row('chat_sessions', {'id': session_id}, {'title': smart_title, 'updated_at': now})

        # schedule FAISS addition (best-effort)
        try:
            await faiss_add(message, reply_text)
        except Exception:
            logger.exception("FAISS add failed.")
    except Exception:
        logger.exception("Supabase save failed during normal chat.")

    return {
        "reply": reply_text,
        "in_flow": False,
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply_text}
        ],
    }

@router.get("/search_chats/{user_id}")
async def search_chats(user_id: str, q: str = Query(..., min_length=1)):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        r = supabase.table('chat_sessions').select('*').ilike('title', f'%{q}%').eq('user_id', user_id).order('updated_at', desc=True).execute()
        return r.data or []
    except Exception:
        logger.exception('search_chats failed')
        raise HTTPException(status_code=500, detail='Search failed')

@router.delete("/clear_chat/{chat_id}")
async def clear_chat(chat_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        delete_rows('chat_messages', {'session_id': chat_id})
        update_row('chat_sessions', {'id': chat_id}, {'title': 'New Chat', 'updated_at': datetime.utcnow().isoformat()})
        return {'success': True}
    except Exception:
        logger.exception('clear_chat failed')
        raise HTTPException(status_code=500, detail='Failed to clear chat')

@router.post("/logout/{user_id}")
async def logout(user_id: str):
    try:
        if user_id in user_sessions:
            del user_sessions[user_id]
        logger.info(f"Logout successful for user {user_id}")
        return {"success": True}
    except Exception as e:
        logger.error(f"logout failed: {e}")
        return {"success": False, "error": str(e)}
