import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.core.db import supabase

logger = logging.getLogger(__name__)


async def save_user_message_immediately(
    session_id: str,
    user_id: str,
    message: str,
):
    if not supabase:
        return

    try:
        supabase.table("chat_messages").insert(
            {
                "session_id": session_id,
                "role": "user",
                "content": message,
                "created_at": datetime.utcnow().isoformat(),
            }
        ).execute()
    except Exception:
        logger.exception("Failed to save user message immediately.")


async def persist_chat_pair(
    user_id: str,
    session_key: str,
    session: Dict[str, Any],
    session_id: Optional[str],
    user_message: Optional[str],
    assistant_message: Optional[str],
    attachment: Optional[Dict[str, Any]] = None,
    user_attachment: Optional[Dict[str, Any]] = None,
):
    if not supabase or not session_id:
        return session_id

    try:
        now = datetime.utcnow().isoformat()
        user_inserted_at: Optional[datetime] = None

        supabase.table("chat_sessions").update({"updated_at": now}).eq("id", session_id).execute()

        user_text = user_message.strip() if isinstance(user_message, str) else ""
        if user_text or user_attachment:
            user_inserted_at = datetime.utcnow()
            payload = {
                "session_id": session_id,
                "role": "user",
                "content": user_text,
                "created_at": user_inserted_at.isoformat(),
            }
            if user_attachment:
                payload["attachment"] = user_attachment
            supabase.table("chat_messages").insert(payload).execute()

        if assistant_message is not None:
            text = assistant_message.strip()
            if text:
                assistant_time = datetime.utcnow()
                if user_inserted_at is not None and assistant_time <= user_inserted_at:
                    assistant_time = user_inserted_at + timedelta(microseconds=1)
                msg_payload = {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": text,
                    "created_at": assistant_time.isoformat(),
                }

                if attachment:
                    msg_payload["attachment"] = attachment

                supabase.table("chat_messages").insert(msg_payload).execute()

        return session_id

    except Exception:
        logger.exception("Failed to persist chat messages.")
        return session_id


def generate_instant_smart_title(msg: str, in_flow: bool = False) -> str:
    if not msg:
        return "New Chat"

    text = msg.strip()

    if in_flow:
        return "Service Requirements"

    text = text.rstrip("?.! ")

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

    if text.lower().startswith(("how", "what", "why", "can", "does", "do ")):
        words = text.split()
        if len(words) <= 6:
            return text.capitalize()
        remove = {"how", "what", "why", "can", "do", "does", "is", "the", "a", "to"}
        cleaned = [w for w in words if w.lower() not in remove]
        title = " ".join(cleaned[:6])
        return title.capitalize()

    words = text.split()
    if len(words) > 7:
        words = words[:7]

    title = " ".join(words)
    return title.capitalize()


def update_session_title_if_needed(
    session: dict,
    session_id: str,
    message: str,
    in_flow: bool,
):
    if session.get("title_set"):
        return

    if not supabase or not session_id or not message:
        return

    title = generate_instant_smart_title(message, in_flow=in_flow)

    try:
        supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()
        session["title_set"] = True
    except Exception:
        logger.exception("Failed to update chat title.")
