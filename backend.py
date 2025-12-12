import os
import re
import pickle
import faiss
import difflib
import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("swarai.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Environment Setup
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["LLAMA_CPP_USE_MLOCK"] = "1"
os.environ["LLAMA_CPP_USE_MMAP"] = "1"

FAISS_DIR = os.getenv("FAISS_DIR", "faiss_db")
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Supabase operations will fail until configured.")
    supabase: Optional[Client] = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Globals
embed_model: Optional[SentenceTransformer] = None
llm: Optional[Any] = None
embedding_cache: Dict[str, Any] = {}
_model_lock = asyncio.Lock()
_faiss_lock = asyncio.Lock()

# in-memory FAISS cache
_index_in_memory: Optional[faiss.Index] = None
_faiss_mtime: Optional[float] = None
_chunks_cache: List[str] = []
_sources_cache: List[str] = []


# Prompt template
PROMPT_SYSTEM_GUIDELINES = """
You are SwarAI — a professional chatbot specializing in Windows audio development, software architecture, and conversational AI.

Rules:
- NEVER output <think> or </think> tags.
- NEVER reveal chain-of-thought or hidden reasoning. Only provide the final answer.
- Use short paragraphs for clarity.
- Use bullet points when helpful.
- ALWAYS format code using standard markdown fenced blocks, for example:

```csharp
class Example {
}
"""

PROMPT_TEMPLATE = """{system_guidelines}

Context:
{context}

User question:
{user_input}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["system_guidelines", "context", "user_input"],
    template=PROMPT_TEMPLATE
)

# Utilities
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def matches_services_trigger(text: str) -> bool:
    if not text:
        return False

    text_n = normalize_text(text)

    # Exact phrases
    main_phrases = [
        "try your service",
        "try your services",
        "try service",
        "try services",
        "start service flow",
    ]

    # Acceptable typos
    service_typos = [
        "sevice",
        "servise",
        "serivce",
        "srvice",
        "srevice",
    ]

    # Check multi-word triggers
    for p in main_phrases:
        if p in text_n:
            return True

    # Check typo patterns after "try your"
    if text_n.startswith("try your "):
        for typo in service_typos:
            if typo in text_n:
                return True

    # Strict regex
    if re.search(r"\btry\b.*\bservice(s)?\b", text_n):
        return True

    return False


def split_text_safely(text: str, max_len: int = 1500) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip() if cur else s
        else:
            parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts

def _build_prompt(context_text: str, user_input: str) -> str:
    return prompt_template.format(
        system_guidelines=PROMPT_SYSTEM_GUIDELINES,
        context=context_text,
        user_input=user_input
    )

# Token estimation (simple)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))

# Model + FAISS loading
async def _async_load_models_and_index():
    global embed_model, llm, _index_in_memory, _faiss_mtime, _chunks_cache, _sources_cache
    async with _model_lock:
        if embed_model is not None and llm is not None and _index_in_memory is not None:
            logger.info("Models and FAISS already loaded in memory.")
            return

        logger.info("Loading models and FAISS (async)...")
        try:
            os.makedirs(FAISS_DIR, exist_ok=True)
            if embed_model is None:
                embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer loaded.")

            if llm is None:
                # Download model artifact once (may take time). Adjust repo / filename as needed.
                model_path = hf_hub_download(
                    repo_id=os.getenv("LLAMA_REPO"),
                    filename=os.getenv("LLAMA_FILENAME")
                )
                llm = Llama(model_path=model_path, n_ctx=int(os.getenv("LLM_N_CTX", 2024)), n_batch=int(os.getenv("LLM_N_BATCH", 64)), chat_format="llama-3")
                logger.info("Llama model loaded.")

            # Load or create FAISS index (keep in memory)
            if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
                try:
                    _index_in_memory = faiss.read_index(INDEX_FILE)
                    with open(METADATA_FILE, "rb") as f:
                        metadata = pickle.load(f)
                    _chunks_cache = metadata.get("chunks", [])
                    _sources_cache = metadata.get("sources", [])
                    _faiss_mtime = os.path.getmtime(INDEX_FILE)
                    logger.info("Loaded FAISS index and metadata (entries=%d).", len(_chunks_cache))
                except Exception as e:
                    logger.exception("Error loading FAISS index; creating a new one. Error: %s", e)
                    _index_in_memory = faiss.IndexFlatL2(384)
                    _chunks_cache, _sources_cache = [], []
                    _faiss_mtime = None
            else:
                _index_in_memory = faiss.IndexFlatL2(384)
                _chunks_cache, _sources_cache = [], []
                _faiss_mtime = None
                logger.info("Created new FAISS index in memory.")
        except Exception:
            logger.exception("Failed to load models or FAISS index.")
            raise

def load_models_if_needed():
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and not loop.is_closed():
        # Running inside an event loop; schedule a task if not already loaded.
        asyncio.create_task(_async_load_models_and_index())
    else:
        # No running loop (e.g., called from sync startup). Run the async loader.
        asyncio.run(_async_load_models_and_index())

# Embedding + retrieval
def get_embedding(text: str):
    if text is None:
        return None
    if embed_model is None:
        load_models_if_needed()
    if text not in embedding_cache:
        emb = embed_model.encode([text]).astype("float32")
        embedding_cache[text] = emb
    return embedding_cache[text]

async def get_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[str, str]]:
    global _index_in_memory, _faiss_mtime, _chunks_cache, _sources_cache
    await _ensure_index_loaded()

    if _index_in_memory is None or not _chunks_cache:
        return []

    try:
        q_emb = get_embedding(query)
        if q_emb is None:
            return []
        D, I = _index_in_memory.search(q_emb, min(top_k, len(_chunks_cache)))
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(_chunks_cache):
                continue
            results.append((_chunks_cache[idx], _sources_cache[idx] if idx < len(_sources_cache) else "unknown"))
        return results
    except Exception:
        logger.exception("FAISS retrieval failed for query: %s", query)
        return []

async def _ensure_index_loaded():
    global _index_in_memory, _faiss_mtime, _chunks_cache, _sources_cache
    async with _faiss_lock:
        if _index_in_memory is None:
            await _async_load_models_and_index()
            return

        if os.path.exists(INDEX_FILE):
            try:
                mtime = os.path.getmtime(INDEX_FILE)
                if _faiss_mtime is None or mtime > _faiss_mtime:
                    # reload index + metadata
                    _index_in_memory = faiss.read_index(INDEX_FILE)
                    with open(METADATA_FILE, "rb") as f:
                        metadata = pickle.load(f)
                    _chunks_cache = metadata.get("chunks", [])
                    _sources_cache = metadata.get("sources", [])
                    _faiss_mtime = mtime
                    logger.info("FAISS index reloaded from disk (entries=%d).", len(_chunks_cache))
            except Exception:
                logger.exception("Error reloading FAISS from disk; keeping in-memory index.")

async def add_chat_to_faiss(query: str, response: str):
    global _index_in_memory, _chunks_cache, _sources_cache, _faiss_mtime
    if not query or not response:
        return
    await _ensure_index_loaded()
    async with _faiss_lock:
        try:
            if query in _chunks_cache:
                logger.debug("Query already in FAISS chunks; skipping: %s", query[:40])
                return
            emb = embed_model.encode([query]).astype("float32")
            _index_in_memory.add(emb)
            _chunks_cache.append(response)
            _sources_cache.append("supabase_chat")

            # persist index and metadata atomically
            faiss.write_index(_index_in_memory, INDEX_FILE)
            with open(METADATA_FILE, "wb") as f:
                pickle.dump({"chunks": _chunks_cache, "sources": _sources_cache}, f)
            _faiss_mtime = os.path.getmtime(INDEX_FILE)
            logger.info("Added chat to FAISS and persisted (query=%s...) entries=%d", query[:40], len(_chunks_cache))
        except Exception:
            logger.exception("Failed to update FAISS with new chat.")

# Supabase sync helper
def sync_supabase_history_to_faiss():
    if supabase is None:
        logger.warning("Supabase client not configured; cannot sync history.")
        return
    try:
        result = supabase.table("chat_messages").select("content").execute()
        rows = result.data or []
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for row in rows:
            content = row.get("content")
            if not content:
                continue
            if loop and not loop.is_closed():
                asyncio.create_task(add_chat_to_faiss(content, content))
            else:
                asyncio.run(add_chat_to_faiss(content, content))
        logger.info("Scheduled sync of %d messages from Supabase to FAISS.", len(rows))
    except Exception:
        logger.exception("Error syncing Supabase history to FAISS.")

async def _call_model_async(prompt_text: str, max_tokens: int = 512) -> str:
    await _async_load_models_and_index()  # ensure loaded
    async with _model_lock:
        try:
            def _run():
                return llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": PROMPT_SYSTEM_GUIDELINES},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=max_tokens,
                )

            result = await asyncio.to_thread(_run)

            text = result["choices"][0]["message"]["content"].strip()
            return text

        except Exception:
            logger.exception("Model call failed.")
            raise

async def generate_answer_async(query: str) -> List[str]:
    if not query:
        return ["Sorry — empty query received."]
    try:
        results = await get_similar_chunks(query, top_k=5)
        context_text = "\n\n".join([c for c, _ in results]) if results else ""
        if len(context_text) > 4000:
            context_text = context_text[:4000]
        prompt_text = _build_prompt(context_text, query)
        raw = await _call_model_async(prompt_text, max_tokens=512)
        parts = split_text_safely(raw, max_len=1500)

        try:
            asyncio.create_task(add_chat_to_faiss(query, raw))
        except Exception:
            logger.exception("Failed to schedule FAISS add task.")

        return parts
    except Exception:
        logger.exception("Generation failed.")
        return ["Sorry, something went wrong during generation."]

def generate_answer(query: str) -> List[str]:
    return asyncio.get_event_loop().run_until_complete(generate_answer_async(query))
