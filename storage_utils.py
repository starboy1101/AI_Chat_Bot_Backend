from supabase import create_client
from uuid import uuid4
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_pdf_to_supabase(local_path: str, filename: str) -> str:
    with open(local_path, "rb") as f:
        supabase.storage.from_("chat-files").upload(
            filename,
            f,
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


