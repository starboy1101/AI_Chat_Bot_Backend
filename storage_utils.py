from supabase import create_client
from uuid import uuid4
from typing import Union
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_pdf_to_supabase(pdf_bytes: bytes, filename: str) -> str:
    supabase.storage.from_("chat-files").upload(
        filename,
        pdf_bytes,
        {
            "content-type": "application/pdf",
            "upsert": "true",
        },
    )

    return (
        supabase.storage
        .from_("chat-files")
        .get_public_url(filename)
    )


