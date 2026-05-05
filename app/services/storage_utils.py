import os

from app.core.db import supabase

_CONTENT_TYPES_BY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def upload_file_to_supabase(file_bytes: bytes, filename: str, content_type: str | None = None) -> str:
    if not file_bytes:
        raise ValueError("Empty file upload")
    if not supabase:
        raise RuntimeError("Supabase not configured")

    ext = os.path.splitext((filename or "").lower())[1]
    resolved_content_type = content_type or _CONTENT_TYPES_BY_EXT.get(ext, "application/octet-stream")

    supabase.storage.from_("chat-files").upload(
        filename,
        file_bytes,
        {
            "content-type": resolved_content_type,
            "upsert": "true",
        },
    )

    return supabase.storage.from_("chat-files").get_public_url(filename)


def upload_pdf_to_supabase(pdf_bytes: bytes, filename: str) -> str:
    return upload_file_to_supabase(pdf_bytes, filename, content_type="application/pdf")
