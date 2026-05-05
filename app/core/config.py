import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_name: str, default: Path) -> str:
    raw = os.getenv(env_name)
    if not raw:
        return str(default)

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "change_me_replace")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
FAISS_DIR = _resolve_path("FAISS_DIR", PROJECT_ROOT / "faiss_db")
DATA_DIR = _resolve_path("DATA_DIR", PROJECT_ROOT / "data")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2024"))

LLAMA_REPO = os.getenv("LLAMA_REPO")
LLAMA_FILENAME = os.getenv("LLAMA_FILENAME")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
FLOW_FILE = _resolve_path("FLOW_FILE", PROJECT_ROOT / "questions.json")

# PDF LOGIC
PDF_MAX_SIZE_MB = int(os.getenv("PDF_MAX_SIZE_MB", "10"))
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "50"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "32768"))
