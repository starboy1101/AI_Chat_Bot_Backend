import os
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
from db import supabase, insert_row, delete_rows, update_row
from models import ChatRequest, CreateChatRequest, UpdateUserInfo
from llm_manager import generate_answer_async
from flows import flow_manager, conversation_flow
from greetings import is_greeting, GREETING_REPLY
from utils import normalize_text
from config import MAX_CONTEXT_TOKENS
from faiss_manager import add as faiss_add
from pdf_generator import generate_final_requirements_pdf
from storage_utils import upload_pdf_to_supabase

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

async def save_user_message_immediately(session_id, user_id, message):
    if not supabase or not session_id:
        return

    now = datetime.utcnow().isoformat()

    insert_row(
        "chat_messages",
        {
            "session_id": session_id,
            "role": "user",
            "content": message,
            "created_at": now,
        },
    )

    # also update chat session timestamp
    update_row(
        "chat_sessions",
        {"id": session_id},
        {"updated_at": now},
    )

@router.get("/userinfo/{user_id}")
async def get_user_info(user_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')

    try:
        result = (
            supabase.table('users')
            .select("user_id, email, first_name, last_name, bio, location, website")
            .eq('user_id', user_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "success": True,
            "data": result.data[0]
        }

    except Exception:
        logger.exception("Failed to fetch user info")
        raise HTTPException(status_code=500, detail="Failed to fetch user info")

@router.post("/userinfo/update")
async def update_user_info(req: UpdateUserInfo):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        result = (
            supabase.table("users")
            .update({
                "first_name": req.first_name,
                "last_name": req.last_name,
                "email": req.email,
                "bio": req.bio,
                "location": req.location,
                "website": req.website,
            })
            .eq("user_id", req.user_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "success": True,
            "message": "User info updated successfully",
            "data": result.data[0]
        }

    except Exception as e:
        logger.exception("Failed to update user info")
        raise HTTPException(status_code=500, detail="Failed to update user info")


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
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        r = (
            supabase
            .table("chat_messages")
            .select("id, role, content, attachment, created_at")
            .eq("session_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )
        return r.data or []

    except Exception:
        logger.exception("get_chat failed")
        raise HTTPException(status_code=500, detail="Failed to fetch chat messages")


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

def generate_instant_smart_title(msg: str, in_flow: bool = False) -> str:
    if not msg:
        return "New Chat"

    text = msg.strip()

    # 1. Flow-based override
    if in_flow:
        return "Service Requirements"

    # 2. Remove trailing punctuation
    text = text.rstrip("?.! ")

    # 3. Audio-domain keyword mapping
    audio_keywords = {
        "sample rate": "Audio Sampling",
        "bit depth": "Audio Bit Depth",
        "equalizer": "Audio Equalizer",
        "optimization": "Audio Optimization",
        "dsp": "DSP Processing",
        "hifi3": "HiFi3 Optimization",
        "hifi4": "HiFi4 Optimization",
        "hifi5": "HiFi5 Optimization",
        "framework": "Audio Framework",
        "intrinsic": "Intrinsic Optimization",
        "porting": "Audio Porting",
        "hexagon": "Hexagon DSP",
    }

    lower_msg = text.lower()
    for key, title in audio_keywords.items():
        if key in lower_msg:
            return title

    # 4. If it's a question → make a clean title
    if text.lower().startswith(("how", "what", "why", "can", "does", "do ")):
        words = text.split()
        if len(words) <= 6:
            return text.capitalize()
        else:
            # Remove filler words
            remove = {"how", "what", "why", "can", "do", "does", "is", "the", "a", "to"}
            cleaned = [w for w in words if w.lower() not in remove]
            title = " ".join(cleaned[:6])
            return title.capitalize()

    # 5. If it's long → extract first 5–7 meaningful words
    words = text.split()
    if len(words) > 7:
        words = words[:7]

    title = " ".join(words)
    return title.capitalize()


async def persist_chat_pair(user_id, session_key, session, session_id, user_message, assistant_message, attachment=None):
    if not supabase or not session_id:
        return session_id

    now = datetime.utcnow().isoformat()

    try:
        insert_row(
            "chat_messages",
            {
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_message,
                "attachment": attachment,
                "created_at": now,
            },
        )

        update_row(
            "chat_sessions",
            {"id": session_id},
            {"updated_at": now},
        )
    except Exception:
        logger.exception("Failed to save assistant message")
        return session_id

    try:
        session_row = (
            supabase.table("chat_sessions")
            .select("title")
            .eq("id", session_id)
            .execute()
        )
        current_title = session_row.data[0]["title"] if session_row.data else "New Chat"

        if current_title == "New Chat" and user_message.strip():
            smart_title = generate_instant_smart_title(
                user_message,
                in_flow=session.get("in_flow", False)
            )

            update_row(
                "chat_sessions",
                {"id": session_id},
                {"title": smart_title, "updated_at": now},
            )

    except Exception:
        logger.exception("Smart title generation failed")

    return session_id


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

    # session_id may or may not be provided by frontend
    raw_session_id = req.session_id or None

    # If session_id is provided → per-chat flow isolate via composite key.
    # If not provided → fall back to per-user key (same as old behavior).
    if raw_session_id:
        session_key = f"{user_id}:{raw_session_id}"
    else:
        session_key = user_id

    # prepare in-memory session defaults (preserve original behavior)
    default_session = {
        "in_flow": False,
        "node_id": None,
        "context": {},
        "change_target": None,
        "authenticated": False,
        "session_id": raw_session_id,
    }
    session = user_sessions.get(session_key, {})
    for k, v in default_session.items():
        session.setdefault(k, v)
    user_sessions[session_key] = session
    is_demo_user = session.get("authenticated", False)  # kept for compatibilit

    async def _get_reply_text(text: str) -> str:
        parts = await generate_answer_async(text)
        if isinstance(parts, list) and parts:
            return parts[0]
        return parts if isinstance(parts, str) else str(parts)

    session_id = session.get("session_id") or raw_session_id

    # Ensure session_id exists before saving user message
    if not session_id and not is_guest and supabase:
        now = datetime.utcnow().isoformat()
        session_id = str(uuid4())
        insert_row(
            "chat_sessions",
            {
                "id": session_id,
                "user_id": user_id,
                "title": "New Chat",
                "created_at": now,
                "updated_at": now,
            },
        )
        session["session_id"] = session_id
        user_sessions[session_key] = session

    # --- Save USER message instantly before any flow or AI ---
    if session_id:
        try:
            await save_user_message_immediately(session_id, user_id, message)
        except Exception:
            logger.exception("Failed to save user message immediately.")    


    # 1) Flow trigger: if not in flow and message triggers service flow (use flow trigger detection from backend.py)
    # backend.py defines matches_services_trigger -- import robustly
    try:
        from backend import matches_services_trigger
    except Exception:
        try:
            from .backend import matches_services_trigger
        except Exception:
            matches_services_trigger = lambda t: False

    logger.info("chat_endpoint: session_key=%s in_flow=%s node_id=%s message=%r",
    session_key, session.get("in_flow"), session.get("node_id"), message)


    if not session["in_flow"] and matches_services_trigger(message):
        session.update({"in_flow": True, "node_id": "start", "context": {}, "change_target": None})
        user_sessions[session_key] = session 
        start_node = conversation_flow.get("start", {"text": "Starting..."})
        reply_text = start_node.get("text", "Starting...")
        options = start_node.get("options", [])
        try:
            session_id = await persist_chat_pair(
                user_id, session_key, session, session_id, message, reply_text
            )
        except Exception:
            logger.exception("Supabase save failed (flow start).")
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

    # 2) If in flow -> delegate to FlowManager logic 
    if session["in_flow"]:
        current_id = session["node_id"]
        node = conversation_flow.get(current_id)

        # ---------------------------------------------
        # HANDLE YES/NO AFTER INVALID OPTION
        # ---------------------------------------------
        if current_id == "mistake_prompt":
            ans = message.strip().lower()

            # YES → return to the node before the mistake
            if ans in ["yes", "y"]:
                last_q = session.get("last_question_node")

                if last_q and last_q in conversation_flow:
                    session["node_id"] = last_q
                    user_sessions[session_key] = session

                    q_node = conversation_flow[last_q]
                    reply_text = q_node.get("text", "Let's continue.")
                    options = q_node.get("options", [])

                    return {
                        "reply": reply_text,
                        "options": options,
                        "in_flow": True,
                        "node_id": last_q,
                        "context": session["context"],
                    }

                # fallback if node is missing
                session["in_flow"] = False
                user_sessions[session_key] = session
                reply_text = "⚠️ Unable to continue because the previous step was not found."
                return {"reply": reply_text, "in_flow": False}

            # NO → end flow
            if ans in ["no", "n"]:
                session["in_flow"] = False
                session["node_id"] = None
                user_sessions[session_key] = session

                reply_text = "👍 No problem — ending the requirements flow."
                return {
                    "reply": reply_text,
                    "options": [],
                    "in_flow": False,
                    "node_id": None,
                    "context": session["context"],
                }

            # Invalid yes/no → ask again
            reply_text = "Please type **Yes** or **No**."
            return {
                "reply": reply_text,
                "options": ["Yes", "No"],
                "in_flow": True,
                "node_id": "mistake_prompt",
                "context": session["context"],
            }

        # If node missing → exit flow safely
        if not node:
            session["in_flow"] = False
            user_sessions[session_key] = session
            safe_reply = await _get_reply_text(req.message)
            try:
                session_id = await persist_chat_pair(
                    user_id, session_key, session, session_id, message, safe_reply
                )
            except Exception:
                logger.exception("Supabase save failed (flow missing node).")
            return {
                "reply": safe_reply,
                "in_flow": False,
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": safe_reply}
                ],
            }

        # ---------------------------------------------
        # HANDLE MAKE_CHANGES (expects number)
        # ---------------------------------------------
        if current_id == "make_changes" and node.get("expect_user_input"):
            try:
                change_num = int(message.strip())
                summary_items = list(session["context"].keys())

                if change_num < 1 or change_num > len(summary_items):
                    raise ValueError("Invalid number")

                target_id = summary_items[change_num - 1]
                session["change_target"] = target_id

                # next node
                target_node = conversation_flow.get(target_id, {})
                question_text = target_node.get("text", "")
                options = target_node.get("options", [])

                if options:
                    reply_text = f"📝 Let's update your answer!\n\n{question_text}"
                    session["node_id"] = target_id
                else:
                    # move to await_new_answer
                    reply_text = f"{conversation_flow['await_new_answer']['text']}\n\n📝 Current Question: {question_text}"
                    session["node_id"] = "await_new_answer"

                user_sessions[session_key] = session
                try:
                    session_id = await persist_chat_pair(
                        user_id, session_key, session, session_id, message, reply_text
                    )
                except:
                    logger.exception("Supabase save failed (make_changes).")

                return {
                    "reply": reply_text,
                    "options": options,
                    "in_flow": True,
                    "node_id": session["node_id"],
                    "context": session["context"],
                }

            except Exception:
                reply_text = "⚠️ Please enter a valid question number (e.g., 1, 2, 3)."
                try:
                    session_id = await persist_chat_pair(
                        user_id, session_key, session, session_id, message, reply_text
                    )
                except:
                    logger.exception("Supabase save failed (invalid change number).")

                return {
                    "reply": reply_text,
                    "options": [],
                    "in_flow": True,
                    "node_id": "make_changes",
                    "context": session["context"],
                }

        # ---------------------------------------------
        # HANDLE AWAIT_NEW_ANSWER
        # ---------------------------------------------
        if current_id == "await_new_answer" and node.get("expect_user_input"):
            target_id = session.get("change_target")

            if target_id and target_id in session["context"]:
                session["context"][target_id] = message.strip()
                session["change_target"] = None
                session["node_id"] = "show_updated_summary"
                user_sessions[session_key] = session

                updated_summary = "📝 Updated Summary:\n\n"
                for idx, (key, val) in enumerate(session["context"].items(), start=1):
                    q_text = conversation_flow[key].get("text", "")
                    updated_summary += f"{idx}. {q_text} → {val}\n"

                reply_text = conversation_flow["show_updated_summary"]["text"].replace(
                    "{{updated_summary}}", updated_summary
                )
                options = conversation_flow["show_updated_summary"].get("options", [])

                try:
                    session_id = await persist_chat_pair(
                        user_id, session_key, session, session_id, message, reply_text
                    )
                except:
                    logger.exception("Supabase save failed (await_new_answer).")

                return {
                    "reply": reply_text,
                    "options": options,
                    "in_flow": True,
                    "node_id": "show_updated_summary",
                    "context": session["context"],
                }

        # ---------------------------------------------
        # GENERAL EXPECT_USER_INPUT
        # ---------------------------------------------
        if node.get("expect_user_input") or node.get("type") == "input":
            node_id = node.get("id") or current_id

            # ✅ FIX: accumulate multiple queries
            if node_id == "query_input":
                session["context"].setdefault("query_input", []).append(message.strip())
            else:
                session["context"][node_id] = message.strip()

            session["node_id"] = node.get("next")
            user_sessions[session_key] = session
            
            next_node = conversation_flow.get(node["next"])
            if not next_node:
                # fall back to normal model response
                session["in_flow"] = False
                user_sessions[session_key] = session
                safe_reply = await _get_reply_text(req.message)

                try:
                    session_id = await persist_chat_pair(
                        user_id, session_key, session, session_id, message, safe_reply
                    )
                except:
                    logger.exception("Supabase save failed (fallback input).")

                return {"reply": safe_reply, "in_flow": False}

            reply_text = next_node["text"]
            options = next_node.get("options", [])

            try:
                session_id = await persist_chat_pair(
                    user_id, session_key, session, session_id, message, reply_text
                )
            except:
                logger.exception("Supabase save failed (expect_user_input).")

            return {
                "reply": reply_text,
                "options": options,
                "in_flow": True,
                "node_id": node["next"],
                "context": session["context"],
            }

        # ---------------------------------------------
        # OPTION SELECTION (THIS WAS BROKEN BEFORE)
        # ---------------------------------------------
        if "options" in node and node["options"]:
            selected = flow_manager.find_best_option(node["options"], message)

            if not selected:
                mistake = conversation_flow["mistake_prompt"]
                reply_text = "⚠️ That didn’t match any available option.\n\n Would you like to continue the requirements?"

                # Move to a special pseudo-state inside the same node
                session["node_id"] = "mistake_prompt"
                session["last_question_node"] = current_id
                user_sessions[session_key] = session

                try:
                    session_id = await persist_chat_pair(
                        user_id, session_key, session, session_id, message, reply_text
                    )
                except:
                    logger.exception("Supabase save failed (mistake_prompt).")

                return {
                    "reply": reply_text,
                    "options": mistake.get("options", []),
                    "in_flow": True,
                    "node_id": "mistake_prompt",
                    "context": session["context"],
                }

            session["context"][node["id"]] = selected["label"]
            next_id = selected.get("next")

        else:
            next_id = node.get("next")

        # ---------------------------------------------
        # MOVE TO NEXT NODE
        # ---------------------------------------------
        next_node = conversation_flow.get(next_id)

        if not next_node:
            session["in_flow"] = False
            user_sessions[session_key] = session
            safe_reply = await _get_reply_text(req.message)

            try:
                session_id = await persist_chat_pair(
                    user_id, session_key, session, session_id, message, safe_reply
                )
            except:
                logger.exception("Supabase save failed (no next node).")

            return {"reply": safe_reply, "in_flow": False}

        session["node_id"] = next_id
        session["in_flow"] = (next_id != "submit_response")
        user_sessions[session_key] = session

        # ---------------- FINAL STEP ------------------
        if next_id == "submit_response":
            pdf_filename = f"REQ_{session_id}.pdf"
            pdf_path = os.path.join("generated_pdfs", pdf_filename)

            generate_final_requirements_pdf(
                context=session["context"],
                path=pdf_path
            )

            pdf_url = upload_pdf_to_supabase(pdf_path, pdf_filename)

            reply_text = (
                "Thank you! Your requirements have been submitted. "
                "Our team will get back to you shortly.\n\n"
                "📄 Please find the attached requirements document below."
            )

            attachment = {
                "type": "pdf",
                "name": "Requirements.pdf",
                "url": pdf_url,
            }

            try:
                session_id = await persist_chat_pair(
                    user_id, session_key, session, session_id, message, reply_text,  attachment=attachment
                )
            except:
                logger.exception("Supabase save failed (submit_response).")

            return {
                "reply": reply_text,
                "in_flow": False,
                "node_id": "submit_response",
                "attachment": attachment,
                "context": session["context"],
            }

        # -------- NORMAL FLOW (NON-FINAL) -------------
        if next_id == "show_summary":
            summary_text = "📝 Summary of your responses:\n\n"
            for idx, (k, v) in enumerate(session["context"].items(), start=1):
                q = conversation_flow[k].get("text", "")
                summary_text += f"{idx}. {q} → {v}\n"

            reply_text = next_node["text"].replace("{{summary}}", summary_text)
        else:
            reply_text = next_node["text"]

        options = next_node.get("options", [])

        try:
            session_id = await persist_chat_pair(
                user_id, session_key, session, session_id, message, reply_text
            )
        except:
            logger.exception("Supabase save failed (next node).")

        return {
            "reply": reply_text,
            "options": options,
            "in_flow": True,
            "node_id": next_id,
            "context": session["context"],
        }

    # 3) Greeting handling (robust)
    if is_greeting(message):
        reply_text = GREETING_REPLY
        try:
            session_id = await persist_chat_pair(
                user_id, session_key, session, session_id, message, reply_text
            )
        except Exception:
            logger.exception("Supabase save failed (greeting).")
        return {
            "reply": reply_text,
            "in_flow": False,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply_text},
            ],
        }

    # 4) Standard generation
    reply_text = await _get_reply_text(message)

    # token and context management
    try:
        # use existing or provided session_id for token tracking (can be None key)
        session_id_for_context = req.session_id or session.get("session_id")
        if session_id_for_context not in model_context:
            reset_model_context(session_id_for_context)
        tokens_this_msg = estimate_tokens(message) + estimate_tokens(reply_text)
        update_model_context(session_id_for_context, tokens_this_msg)
        if (
            model_context[session_id_for_context]["tokens_used"]
            > MAX_CONTEXT_TOKENS - 100
        ):
            reset_model_context(session_id_for_context)

        # ensure session exists and persist messages (centralized)
        try:
            session_id = await persist_chat_pair(
                user_id, session_key, session, session_id, message, reply_text
            )
        except Exception:
            logger.exception("Supabase save failed during normal chat (persist).")

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
            {"role": "assistant", "content": reply_text},
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
        # Remove both per-user and per-chat sessions for this user
        keys_to_delete = [
            k
            for k in list(user_sessions.keys())
            if k == user_id or k.startswith(f"{user_id}:")
        ]
        for k in keys_to_delete:
            del user_sessions[k]
        logger.info(f"Logout successful for user {user_id}")
        return {"success": True}
    except Exception as e:
        logger.error(f"logout failed: {e}")
        return {"success": False, "error": str(e)}
