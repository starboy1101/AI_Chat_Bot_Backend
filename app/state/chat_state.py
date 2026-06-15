import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from uuid import uuid4

# shared in-memory session store
user_sessions: Dict[str, Dict[str, Any]] = {}


def _guest_timeout_seconds() -> int:
    raw = os.getenv("GUEST_SESSION_TIMEOUT_SECONDS", str(8 * 60 * 60))
    try:
        return max(60, int(raw))
    except Exception:
        return 8 * 60 * 60


GUEST_SESSION_TIMEOUT_SECONDS = _guest_timeout_seconds()


def is_guest_user(user_id: Optional[str]) -> bool:
    return user_id == "guest" or (isinstance(user_id, str) and user_id.startswith("guest_"))


def _now() -> datetime:
    return datetime.utcnow()


def _iso_now() -> str:
    return _now().isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def guest_index_key(user_id: str) -> str:
    return f"{user_id}:__guest_chat_index__"


def guest_session_key(user_id: str, chat_id: str) -> str:
    return f"{user_id}:{chat_id}"


def _guest_index(user_id: str) -> Dict[str, Any]:
    key = guest_index_key(user_id)
    index = user_sessions.setdefault(
        key,
        {
            "guest_index": True,
            "user_id": user_id,
            "chats": {},
            "last_seen": _iso_now(),
        },
    )
    index.setdefault("chats", {})
    index["last_seen"] = _iso_now()
    return index


def _latest_guest_chat_id(chats: Dict[str, Any]) -> Optional[str]:
    latest_id = None
    latest_updated_at = ""
    for chat_id, chat in chats.items():
        if not isinstance(chat, dict):
            continue
        updated_at = str(chat.get("updated_at", ""))
        if latest_id is None or updated_at > latest_updated_at:
            latest_id = chat_id
            latest_updated_at = updated_at
    return latest_id


def prune_inactive_guest_sessions() -> None:
    cutoff = _now() - timedelta(seconds=GUEST_SESSION_TIMEOUT_SECONDS)

    for key, value in list(user_sessions.items()):
        if not isinstance(value, dict) or not value.get("guest_index"):
            continue

        user_id = value.get("user_id")
        last_seen = _parse_iso(value.get("last_seen"))
        if not isinstance(user_id, str) or not is_guest_user(user_id):
            continue

        chats = value.get("chats")
        if not isinstance(chats, dict):
            chats = {}

        stale_chat_ids = []
        for chat_id, chat in list(chats.items()):
            updated_at = _parse_iso(chat.get("updated_at") if isinstance(chat, dict) else None)
            if updated_at and updated_at < cutoff:
                stale_chat_ids.append(chat_id)

        for chat_id in stale_chat_ids:
            chats.pop(chat_id, None)
            user_sessions.pop(guest_session_key(user_id, chat_id), None)

        if (last_seen and last_seen < cutoff) or not chats:
            for chat_id in list(chats.keys()):
                user_sessions.pop(guest_session_key(user_id, chat_id), None)
            user_sessions.pop(key, None)


def touch_guest_user(user_id: str) -> None:
    if not is_guest_user(user_id):
        return
    prune_inactive_guest_sessions()
    _guest_index(user_id)["last_seen"] = _iso_now()


def ensure_guest_chat(
    user_id: str,
    chat_id: Optional[str] = None,
    title: str = "New Chat",
    create_new: bool = False,
) -> Tuple[str, Dict[str, Any], str]:
    prune_inactive_guest_sessions()
    index = _guest_index(user_id)
    chats = index.setdefault("chats", {})
    resolved_chat_id = chat_id
    if not resolved_chat_id and not create_new:
        resolved_chat_id = _latest_guest_chat_id(chats)
    if not resolved_chat_id:
        resolved_chat_id = str(uuid4())
    now = _iso_now()

    chat = chats.setdefault(
        resolved_chat_id,
        {
            "id": resolved_chat_id,
            "user_id": user_id,
            "title": title or "New Chat",
            "created_at": now,
            "updated_at": now,
            "guest": True,
        },
    )
    chat.setdefault("title", title or "New Chat")
    chat["updated_at"] = chat.get("updated_at") or now

    key = guest_session_key(user_id, resolved_chat_id)
    session = user_sessions.setdefault(
        key,
        {
            "session_id": resolved_chat_id,
            "messages": [],
            "guest": True,
        },
    )
    session.setdefault("session_id", resolved_chat_id)
    session.setdefault("messages", [])
    session["guest"] = True

    index["last_seen"] = now
    return resolved_chat_id, session, key


def append_guest_message(
    user_id: str,
    chat_id: str,
    role: str,
    content: str,
    attachment: Optional[Dict[str, Any]] = None,
) -> None:
    resolved_chat_id, session, _ = ensure_guest_chat(user_id, chat_id)
    now = _iso_now()
    message: Dict[str, Any] = {
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "created_at": now,
    }
    if attachment:
        message["attachment"] = attachment
    session.setdefault("messages", []).append(message)

    index = _guest_index(user_id)
    chat = index.setdefault("chats", {}).setdefault(
        resolved_chat_id,
        {
            "id": resolved_chat_id,
            "user_id": user_id,
            "title": "New Chat",
            "created_at": now,
            "updated_at": now,
            "guest": True,
        },
    )
    chat["updated_at"] = now
    index["last_seen"] = now


def update_guest_chat_title(user_id: str, chat_id: str, title: str) -> None:
    if not title:
        return
    ensure_guest_chat(user_id, chat_id)
    index = _guest_index(user_id)
    chat = index.setdefault("chats", {}).get(chat_id)
    if isinstance(chat, dict):
        chat["title"] = title
        chat["updated_at"] = _iso_now()


def get_guest_chats(user_id: str) -> list[Dict[str, Any]]:
    touch_guest_user(user_id)
    chats = _guest_index(user_id).get("chats", {})
    if not isinstance(chats, dict):
        return []
    return sorted(
        [dict(chat) for chat in chats.values() if isinstance(chat, dict)],
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )


def find_guest_chat(chat_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    prune_inactive_guest_sessions()
    for key, value in list(user_sessions.items()):
        if not key.endswith(f":{chat_id}") or not isinstance(value, dict):
            continue
        user_id = key[: -(len(chat_id) + 1)]
        if is_guest_user(user_id):
            touch_guest_user(user_id)
            return user_id, value, key
    return None, None, None


def get_guest_messages(chat_id: str) -> Optional[list[Dict[str, Any]]]:
    _, session, _ = find_guest_chat(chat_id)
    if not session:
        return None
    return list(session.get("messages", []))


def delete_guest_chat(chat_id: str) -> bool:
    user_id, _, key = find_guest_chat(chat_id)
    if not user_id or not key:
        return False
    index = _guest_index(user_id)
    chats = index.setdefault("chats", {})
    if isinstance(chats, dict):
        chats.pop(chat_id, None)
    user_sessions.pop(key, None)
    return True


def clear_guest_chat(chat_id: str) -> bool:
    user_id, session, _ = find_guest_chat(chat_id)
    if not user_id or not session:
        return False
    session["messages"] = []
    index = _guest_index(user_id)
    chat = index.setdefault("chats", {}).get(chat_id)
    if isinstance(chat, dict):
        chat["title"] = "New Chat"
        chat["updated_at"] = _iso_now()
    session["title_set"] = False
    return True


def search_guest_chats(user_id: str, query: str) -> list[Dict[str, Any]]:
    needle = (query or "").strip().lower()
    return [chat for chat in get_guest_chats(user_id) if needle in chat.get("title", "").lower()]


def delete_guest_user(user_id: str) -> None:
    for key in list(user_sessions.keys()):
        if key == user_id or key.startswith(f"{user_id}:"):
            user_sessions.pop(key, None)
