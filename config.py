from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "change_me_replace")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DEMO_USER = os.getenv("DEMO_USER", "demo_user")
DEMO_PASS = os.getenv("DEMO_PASS", "demo_pass123")
FAISS_DIR = os.getenv("FAISS_DIR", "faiss_db")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2024"))

LLAMA_REPO = os.getenv("LLAMA_REPO")
LLAMA_FILENAME = os.getenv("LLAMA_FILENAME")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
FLOW_FILE = os.getenv("FLOW_FILE", "questions.json")
