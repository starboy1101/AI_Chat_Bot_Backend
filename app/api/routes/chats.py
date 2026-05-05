import os
import logging
import base64
import asyncio
import json
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from app.core.db import supabase, insert_row, delete_rows, update_row
from app.models.schemas import ChatRequest, CreateChatRequest, UpdateUserInfo
from app.services.llm_manager import generate_answer_stream_async
from app.services.backend import matches_services_trigger
from app.utils.greetings import is_greeting, GREETING_REPLY
from app.utils.common import normalize_text
from app.services.faiss_manager import add as faiss_add
from app.services.chat_persistence import (
    persist_chat_pair,
    save_user_message_immediately,
    update_session_title_if_needed,
)
from app.services.chat_handlers import (
    handle_greeting,
    handle_guest_chat,
    handle_flow_engine,
    handle_pdf_upload_and_extraction,
    handle_service_trigger,
    init_or_get_session,
    handle_normal_qa,
)
from app.state.chat_state import user_sessions

router = APIRouter(prefix="/chats")
logger = logging.getLogger("swarai.chats")

# Typing effect pace for character streaming.
def _get_stream_char_delay_seconds() -> float:
    raw = os.getenv("STREAM_CHAR_DELAY_SECONDS", "0.008")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


STREAM_CHAR_DELAY_SECONDS = _get_stream_char_delay_seconds()

# model context tracking (tokens)
model_context: Dict[str, Dict[str, int]] = {}

_DOC_MIME_TYPES = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_TEXT_MIME_TYPES = {"text/plain", "text"}
_SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}

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


def _infer_attachment_kind(attachment: Any) -> Optional[str]:
    if not isinstance(attachment, dict):
        return None

    raw_type = str(attachment.get("type", "")).strip().lower()
    raw_name = str(attachment.get("name", "")).strip().lower()

    if raw_name.endswith(".pdf") or raw_type in {"pdf", "application/pdf"}:
        return "pdf"
    if raw_name.endswith(".docx") or raw_type in {"docx"} or raw_type in _DOC_MIME_TYPES:
        return "docx"
    if raw_name.endswith(".doc") or raw_type in {"doc"}:
        return "doc"
    if raw_name.endswith(".txt") or raw_type in {"txt", "plain", "text"} or raw_type in _TEXT_MIME_TYPES:
        return "txt"

    # Some clients send a generic type while preserving extension in `name`.
    for ext in _SUPPORTED_UPLOAD_EXTENSIONS:
        if raw_name.endswith(ext):
            return ext.lstrip(".")
    return None

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
            .order("id", desc=False)
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


def _sse_data_chunk(token: str) -> str:
    normalized = token.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _sse_event_chunk(event_name: str, payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    lines = serialized.split("\n")
    return f"event: {event_name}\n" + "".join(f"data: {line}\n" for line in lines) + "\n"


def _normalize_option_labels(raw_options: Any) -> List[str]:
    if not isinstance(raw_options, list):
        return []
    labels: List[str] = []
    for opt in raw_options:
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).strip()
        else:
            label = str(opt).strip()
        if label:
            labels.append(label)
    return labels


def _extract_stream_meta(result: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    followup = result.get("followup")
    question = None
    if isinstance(followup, dict):
        raw_question = followup.get("question")
        if isinstance(raw_question, str) and raw_question.strip():
            question = raw_question.strip()

    options = []
    if isinstance(followup, dict):
        options = _normalize_option_labels(followup.get("options", []))
    if not options:
        options = _normalize_option_labels(result.get("options", []))

    payload: Dict[str, Any] = {"kind": "chat_meta"}
    if isinstance(result.get("node_id"), str):
        payload["node_id"] = result["node_id"]
    if isinstance(result.get("in_flow"), bool):
        payload["in_flow"] = result["in_flow"]
    if question:
        payload["question"] = question
    if options:
        payload["options"] = options
    if isinstance(result.get("attachment"), dict):
        payload["attachment"] = result["attachment"]

    # Avoid emitting empty meta payloads.
    if len(payload) == 1:
        return None
    return payload


def _validate_stream_prompt(req: ChatRequest) -> str:
    prompt = (req.message or "").strip()

    if not prompt and not req.attachment:
        raise HTTPException(status_code=400, detail="Empty message")

    if req.attachment:
        if not isinstance(req.attachment, dict):
            raise HTTPException(status_code=400, detail="Invalid attachment payload.")
        kind = _infer_attachment_kind(req.attachment)
        if kind is None:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, DOC, DOCX, and TXT attachments are supported.",
            )
        if not req.attachment.get("bytes"):
            raise HTTPException(status_code=400, detail="Attachment is missing content bytes.")

    return prompt


def _extract_reply_text_for_stream(result: Any) -> str:
    if isinstance(result, dict):
        reply = result.get("reply")
        if isinstance(reply, str):
            return reply.strip()
        if reply is not None:
            return str(reply).strip()

        followup = result.get("followup")
        if isinstance(followup, dict):
            question = followup.get("question")
            if isinstance(question, str):
                return question.strip()
            if question is not None:
                return str(question).strip()
        return ""

    if isinstance(result, str):
        return result
    return ""


async def _stream_text_char_by_char(
    text: str,
    request: Request,
    stream_name: str,
):
    for ch in text:
        if await request.is_disconnected():
            logger.info("Client disconnected from %s.", stream_name)
            return

        yield _sse_data_chunk(ch)
        if STREAM_CHAR_DELAY_SECONDS > 0:
            await asyncio.sleep(STREAM_CHAR_DELAY_SECONDS)
        else:
            await asyncio.sleep(0)


async def _token_stream_generator(req: ChatRequest, request: Request, stream_name: str):
    prompt = (req.message or "").strip()
    try:
        # Preserve existing greeting behavior (session-aware handler + persistence).
        if is_greeting(prompt):
            user_id = req.user_id or "guest"
            is_guest = user_id.startswith("guest_") or user_id == "guest"

            if is_guest:
                guest_session = user_sessions.get(user_id, {"messages": []})
                guest_session["messages"].append({"role": "user", "content": prompt})
                guest_session["messages"].append({"role": "assistant", "content": GREETING_REPLY})
                user_sessions[user_id] = guest_session
                reply_text = GREETING_REPLY
            else:
                result = await _process_chat_request(req)
                meta_payload = _extract_stream_meta(result)
                if meta_payload and not await request.is_disconnected():
                    yield _sse_event_chunk("meta", meta_payload)
                reply_text = _extract_reply_text_for_stream(result)

            if reply_text:
                async for chunk in _stream_text_char_by_char(reply_text, request, stream_name):
                    yield chunk
            return

        user_id = req.user_id or "guest"
        is_guest = user_id.startswith("guest_") or user_id == "guest"
        session = None
        session_key = None
        session_id = None

        if is_guest:
            guest_session = user_sessions.get(user_id, {"messages": []})
            guest_session["messages"].append({"role": "user", "content": prompt})
            user_sessions[user_id] = guest_session
        else:
            session, session_key, session_id = await init_or_get_session(req, user_id)

            # Flow interactions (existing flow or trigger phrase) must run through the
            # regular request processor; otherwise streaming path bypasses flow logic.
            if req.attachment or session.get("in_flow") or matches_services_trigger(prompt):
                result = await _process_chat_request(req)
                meta_payload = _extract_stream_meta(result)
                if meta_payload and not await request.is_disconnected():
                    yield _sse_event_chunk("meta", meta_payload)
                reply_text = _extract_reply_text_for_stream(result)
                if reply_text:
                    async for chunk in _stream_text_char_by_char(reply_text, request, stream_name):
                        yield chunk
                return

            update_session_title_if_needed(
                session=session,
                session_id=session_id,
                message=prompt,
                in_flow=False,
            )
            if session_id:
                await persist_chat_pair(
                    user_id=user_id,
                    session_key=session_key,
                    session=session,
                    session_id=session_id,
                    user_message=prompt,
                    assistant_message=None,
                )

        reply_parts: List[str] = []
        async for token in generate_answer_stream_async(prompt):
            if await request.is_disconnected():
                logger.info("Client disconnected from %s.", stream_name)
                break

            if not token:
                continue

            reply_parts.append(token)
            async for chunk in _stream_text_char_by_char(token, request, stream_name):
                yield chunk

        full_reply = "".join(reply_parts).strip()
        if not full_reply:
            return

        if is_guest:
            guest_session = user_sessions.get(user_id, {"messages": []})
            guest_session["messages"].append({"role": "assistant", "content": full_reply})
            user_sessions[user_id] = guest_session
            return

        if session_id:
            await persist_chat_pair(
                user_id=user_id,
                session_key=session_key,
                session=session,
                session_id=session_id,
                user_message=None,
                assistant_message=full_reply,
            )
    except asyncio.CancelledError:
        logger.info("%s cancelled by server/runtime.", stream_name)
        raise
    except Exception:
        logger.exception("%s failed", stream_name)
        if not await request.is_disconnected():
            yield _sse_data_chunk("[ERROR] Internal server error")


def _stream_response(prompt: str, request: Request, stream_name: str) -> StreamingResponse:
    return StreamingResponse(
        _token_stream_generator(
            ChatRequest(message=prompt),
            request,
            stream_name,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _process_chat_request(req: ChatRequest):
    user_id = req.user_id or "guest"
    message = (req.message or "").strip()

    # ---------------- VALIDATION ----------------
    if not message and not req.attachment:
        raise HTTPException(status_code=400, detail="Empty message")

    # ---------------- GUEST CHAT ----------------
    is_guest = user_id.startswith("guest_") or user_id == "guest"
    if is_guest:
        return await handle_guest_chat(user_id, message)

    # ---------------- SESSION SETUP ----------------
    session, session_key, session_id = await init_or_get_session(req, user_id)

    update_session_title_if_needed(
        session=session,
        session_id=session_id,
        message=message,
        in_flow=False,
    )
    
    # ---------------- GREETING ----------------
    # Do not route attachments or active flow steps to greeting handling.
    if is_greeting(message) and not req.attachment and not session.get("in_flow"):
        return await handle_greeting(
            message=message,
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
        )

    # ---------------- SERVICE TRIGGER ----------------
    if not session["in_flow"] and matches_services_trigger(message):
        return await handle_service_trigger(
                message=message,
                user_id=user_id,
                session=session,
                session_key=session_key,
                session_id=session_id,
            )


    # ---------------- NON-FLOW CHAT ----------------
    if not session["in_flow"]:
        return await handle_normal_qa(
            message=message,
            session=session,
            session_key=session_key,
            user_id=user_id,
            session_id=session_id,
        )

    # ---------------- DOCUMENT UPLOAD + EXTRACTION ----------------
    if (
        session.get("pdf_mode")
        and not session.get("pdf_processed")
        and req.attachment
    ):
        return await handle_pdf_upload_and_extraction(
            req=req,
            session=session,
            session_key=session_key,
            user_id=user_id,
            session_id=session_id,
            message=message,
        )

    # ---------------- FLOW ENGINE ----------------
    return await handle_flow_engine(
        message=message,
        session=session,
        session_key=session_key,
        user_id=user_id,
        session_id=session_id,
    )


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    _validate_stream_prompt(req)
    return StreamingResponse(
        _token_stream_generator(req, request, "/chats/chat stream"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat_sync")
async def chat_sync_endpoint(req: ChatRequest):
    """
    n8n-friendly JSON endpoint.

    The existing chat endpoints stream SSE character chunks, which makes
    downstream automation tools harder to wire visually. This endpoint runs the
    same chat/session/flow logic but returns the final structured JSON payload.
    """
    _validate_stream_prompt(req)
    result = await _process_chat_request(req)
    if isinstance(result, dict):
        return result
    return {
        "reply": str(result),
        "in_flow": False,
        "session_id": req.session_id,
    }


@router.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest, request: Request):
    _validate_stream_prompt(req)
    return StreamingResponse(
        _token_stream_generator(req, request, "/chats/chat_stream stream"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

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
