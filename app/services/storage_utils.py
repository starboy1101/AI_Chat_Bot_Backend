import hashlib
import os
import re

from app.core.db import supabase

_CONTENT_TYPES_BY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_MAX_STORAGE_KEY_LENGTH = 180


def _storage_key_for(filename: str, file_bytes: bytes) -> str:
    raw_name = str(filename or "uploaded_file").strip() or "uploaded_file"
    basename = raw_name.replace("\\", "/").rsplit("/", 1)[-1].strip() or "uploaded_file"
    stem, ext = os.path.splitext(basename)

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "uploaded_file"
    safe_ext = re.sub(r"[^A-Za-z0-9.]+", "", ext.lower())[:24]
    digest = hashlib.sha256(raw_name.encode("utf-8", errors="ignore") + b":" + file_bytes[:4096]).hexdigest()[:12]

    max_stem_length = max(1, _MAX_STORAGE_KEY_LENGTH - len(safe_ext) - len(digest) - 1)
    safe_stem = safe_stem[:max_stem_length].strip("._-") or "uploaded_file"
    return f"{safe_stem}_{digest}{safe_ext}"


def upload_file_to_supabase(file_bytes: bytes, filename: str, content_type: str | None = None) -> str:
    if not file_bytes:
        raise ValueError("Empty file upload")
    if not supabase:
        raise RuntimeError("Supabase not configured")

    storage_key = _storage_key_for(filename, file_bytes)
    ext = os.path.splitext(storage_key.lower())[1]
    resolved_content_type = content_type or _CONTENT_TYPES_BY_EXT.get(ext, "application/octet-stream")

    supabase.storage.from_("chat-files").upload(
        storage_key,
        file_bytes,
        {
            "content-type": resolved_content_type,
            "upsert": "true",
        },
    )

    return supabase.storage.from_("chat-files").get_public_url(storage_key)


def upload_pdf_to_supabase(pdf_bytes: bytes, filename: str) -> str:
    return upload_file_to_supabase(pdf_bytes, filename, content_type="application/pdf")
