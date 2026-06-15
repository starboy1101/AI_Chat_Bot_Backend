import os
import re
import pickle
import faiss
import difflib
import asyncio
import logging
import inspect
import json
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import requests

from app.core.device import default_llama_gpu_layers, sentence_transformer_device
from app.core.config import FAISS_DIR
from app.core.db import supabase

load_dotenv()

logger = logging.getLogger("swarai.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Environment Setup
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["LLAMA_CPP_USE_MLOCK"] = "1"
os.environ["LLAMA_CPP_USE_MMAP"] = "1"

INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")

if supabase is None:
    logger.warning("Supabase client not configured. History sync will stay disabled.")

# Globals
embed_model: Optional[SentenceTransformer] = None
llm: Optional[Any] = None
pdf_llm: Optional[Any] = None
embedding_cache: Dict[str, Any] = {}
_model_lock = asyncio.Lock()
_pdf_model_lock = asyncio.Lock()
_faiss_lock = asyncio.Lock()

# in-memory FAISS cache
_index_in_memory: Optional[faiss.Index] = None
_faiss_mtime: Optional[float] = None
_chunks_cache: List[str] = []
_sources_cache: List[str] = []


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d.", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%r; using %s.", name, raw, default)
    return default


def _resolve_chat_format(name: str, default: str = "chatml") -> Optional[str]:
    raw = os.getenv(name, default).strip()
    if raw.lower() in {"", "auto", "none", "null"}:
        return None
    return raw


def _is_pdf_extractor_configured() -> bool:
    return bool(os.getenv("PDF_EXTRACTOR_REPO") and os.getenv("PDF_EXTRACTOR_FILENAME"))


def _use_llama_cpp_backend() -> bool:
    return _env_bool("USE_LLAMA_CPP_BACKEND", False)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", os.getenv("LOCAL_AI_SRS_BASE_URL", "http://localhost:11434")).rstrip("/")


def _ollama_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", os.getenv("LOCAL_AI_SRS_TEXT_MODEL", "qwen2.5:3b"))


def _ollama_extractor_model() -> str:
    return os.getenv("OLLAMA_EXTRACTOR_MODEL", os.getenv("LOCAL_AI_SRS_HELPER_MODEL", "qwen2.5:3b"))


def _ollama_timeout_seconds() -> int:
    return _env_int("OLLAMA_TIMEOUT_SECONDS", _env_int("LOCAL_AI_SRS_TIMEOUT_SECONDS", 1800))


def _ollama_options(max_tokens: int) -> Dict[str, Any]:
    return {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        "num_ctx": _env_int("OLLAMA_NUM_CTX", _env_int("LOCAL_AI_SRS_NUM_CTX", 4096)),
        "num_predict": max_tokens,
    }


def _ollama_chat_sync(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    stream: bool = False,
):
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": _ollama_options(max_tokens),
    }
    response = requests.post(
        f"{_ollama_base_url()}/api/chat",
        json=payload,
        timeout=_ollama_timeout_seconds(),
        stream=stream,
    )
    response.raise_for_status()
    return response


def _ollama_response_text(response: requests.Response) -> str:
    data = response.json()
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(data.get("response") or "").strip() if isinstance(data, dict) else ""


def _build_llama_instance(
    *,
    model_path: str,
    n_ctx: int,
    n_batch: int,
    n_threads: Optional[int],
    n_threads_batch: Optional[int],
    n_gpu_layers: int,
    chat_format: Optional[str],
    offload_kqv: bool,
    flash_attn: bool,
) -> Any:
    init_params = set(inspect.signature(Llama.__init__).parameters.keys())

    kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
    }
    if "n_gpu_layers" in init_params:
        kwargs["n_gpu_layers"] = n_gpu_layers
    if "offload_kqv" in init_params:
        kwargs["offload_kqv"] = offload_kqv
    if "flash_attn" in init_params:
        kwargs["flash_attn"] = flash_attn
    if n_threads is not None and "n_threads" in init_params:
        kwargs["n_threads"] = n_threads
    if n_threads_batch is not None and "n_threads_batch" in init_params:
        kwargs["n_threads_batch"] = n_threads_batch
    if chat_format is not None and "chat_format" in init_params:
        kwargs["chat_format"] = chat_format
    return Llama(**kwargs)


def _build_llama_with_device_fallback(**kwargs: Any) -> Any:
    n_gpu_layers = int(kwargs.get("n_gpu_layers", 0) or 0)
    if n_gpu_layers == 0:
        logger.info("Loading Llama on CPU.")
        return _build_llama_instance(**kwargs)

    try:
        logger.info("Loading Llama with GPU acceleration (n_gpu_layers=%s).", n_gpu_layers)
        return _build_llama_instance(**kwargs)
    except Exception:
        logger.exception(
            "Failed to load Llama with GPU acceleration. Retrying on CPU. "
            "If you expected GPU support, reinstall llama-cpp-python with CUDA enabled."
        )
        cpu_kwargs = dict(kwargs)
        cpu_kwargs["n_gpu_layers"] = 0
        cpu_kwargs["offload_kqv"] = False
        cpu_kwargs["flash_attn"] = False
        return _build_llama_instance(**cpu_kwargs)


def _load_models_and_index_sync():
    global embed_model, llm, pdf_llm, _index_in_memory, _faiss_mtime, _chunks_cache, _sources_cache

    os.makedirs(FAISS_DIR, exist_ok=True)

    if embed_model is None:
        device = sentence_transformer_device()
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info("SentenceTransformer loaded on %s.", device)

    if _use_llama_cpp_backend() and llm is None:
        # Download model artifact once (may take time). Adjust repo / filename as needed.
        model_path = hf_hub_download(
            repo_id=os.getenv("LLAMA_REPO"),
            filename=os.getenv("LLAMA_FILENAME")
        )
        cpu_threads = max(1, os.cpu_count() or 1)
        llm = _build_llama_with_device_fallback(
            model_path=model_path,
            n_ctx=_env_int("LLM_N_CTX", 2048),
            n_batch=_env_int("LLM_N_BATCH", 128),
            n_threads=_env_int("LLM_N_THREADS", cpu_threads),
            n_threads_batch=_env_int("LLM_N_THREADS_BATCH", cpu_threads),
            n_gpu_layers=_env_int("LLM_N_GPU_LAYERS", default_llama_gpu_layers()),
            chat_format=_resolve_chat_format("LLM_CHAT_FORMAT", "chatml"),
            offload_kqv=_env_bool("LLM_OFFLOAD_KQV", True),
            flash_attn=_env_bool("LLM_FLASH_ATTN", False),
        )
        logger.info("Llama model loaded.")
    elif not _use_llama_cpp_backend():
        logger.info("Skipping llama.cpp LLM load; using Ollama at %s.", _ollama_base_url())

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


def _load_pdf_extractor_model_sync():
    global pdf_llm

    if not _use_llama_cpp_backend() or not _is_pdf_extractor_configured() or pdf_llm is not None:
        return

    pdf_model_path = hf_hub_download(
        repo_id=os.getenv("PDF_EXTRACTOR_REPO"),
        filename=os.getenv("PDF_EXTRACTOR_FILENAME"),
    )
    cpu_threads = max(1, os.cpu_count() or 1)
    pdf_llm = _build_llama_with_device_fallback(
        model_path=pdf_model_path,
        n_ctx=_env_int("PDF_EXTRACTOR_N_CTX", 4096),
        n_batch=_env_int("PDF_EXTRACTOR_N_BATCH", 256),
        n_threads=_env_int("PDF_EXTRACTOR_N_THREADS", _env_int("LLM_N_THREADS", cpu_threads)),
        n_threads_batch=_env_int("PDF_EXTRACTOR_N_THREADS_BATCH", _env_int("LLM_N_THREADS_BATCH", cpu_threads)),
        n_gpu_layers=_env_int(
            "PDF_EXTRACTOR_N_GPU_LAYERS",
            _env_int("LLM_N_GPU_LAYERS", default_llama_gpu_layers()),
        ),
        chat_format=_resolve_chat_format("PDF_EXTRACTOR_CHAT_FORMAT", "chatml"),
        offload_kqv=_env_bool("PDF_EXTRACTOR_OFFLOAD_KQV", _env_bool("LLM_OFFLOAD_KQV", True)),
        flash_attn=_env_bool("PDF_EXTRACTOR_FLASH_ATTN", _env_bool("LLM_FLASH_ATTN", False)),
    )
    logger.info("Dedicated PDF extractor model loaded.")


async def _async_preload_models(preload_pdf: bool = True):
    await _async_load_models_and_index()
    if preload_pdf and _is_pdf_extractor_configured():
        await _async_load_pdf_extractor_model()


async def _async_warmup_models(warmup_pdf: bool = True):
    if _use_llama_cpp_backend() and llm is not None:
        try:
            await _call_model_async("Reply with only: warm", max_tokens=8)
            logger.info("Primary LLM warmup completed.")
        except Exception:
            logger.exception("Primary LLM warmup failed.")
    elif not _use_llama_cpp_backend():
        try:
            await _call_model_async("Reply with only: warm", max_tokens=8)
            logger.info("Ollama chat model warmup completed.")
        except Exception:
            logger.exception("Ollama chat model warmup failed.")

    if warmup_pdf and (_is_pdf_extractor_configured() or not _use_llama_cpp_backend()):
        try:
            await _call_pdf_extractor_async('Return exactly: {"warmup": true}', max_tokens=24)
            logger.info("Extractor model warmup completed.")
        except Exception:
            logger.exception("Extractor model warmup failed.")


async def preload_models_for_startup():
    preload_pdf = _env_bool("PDF_PRELOAD_ON_STARTUP", True)
    warmup_models = _env_bool("LLM_WARMUP_ON_STARTUP", True)

    await _async_preload_models(preload_pdf=preload_pdf)
    if warmup_models:
        await _async_warmup_models(warmup_pdf=preload_pdf)


# Prompt template
PROMPT_SYSTEM_GUIDELINES = """
You are SwarAI, a professional AI assistant.

Rules:
- Provide ONE clear, concise answer only.
- NEVER repeat the answer.
- NEVER include <think> or </think>.
- Do NOT mention internal reasoning.
- If unsure, say you are unsure.
- Prefer correctness over verbosity.
- Format code using markdown fenced blocks only.
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
    if not text_n:
        return False

    # Common service keyword + frequent misspellings.
    if re.search(r"\bservi?c?e?s?\b|\bse?rvi?ce\b|\bsrvice\b|\bsrevice\b", text_n):
        return True

    # Service intents that should also start the requirement flow.
    if re.search(
        r"\bport(?:ing|ed|s)?\b"
        r"|\boptimi[sz](?:e|ed|ing|ation|ations)?\b"
        r"|\baudio\s*(?:app|apps|application|applications)\b",
        text_n,
    ):
        return True

    # Catch simple typos in single-token intents.
    for token in text_n.split():
        if difflib.get_close_matches(token, ["service", "porting", "optimization"], n=1, cutoff=0.86):
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
    global embed_model, llm, pdf_llm, _index_in_memory, _faiss_mtime, _chunks_cache, _sources_cache
    async with _model_lock:
        llm_ready = llm is not None if _use_llama_cpp_backend() else True
        if embed_model is not None and llm_ready and _index_in_memory is not None:
            logger.info("Models and FAISS already loaded in memory.")
            return

        logger.info("Loading models and FAISS (async)...")
        try:
            await asyncio.to_thread(_load_models_and_index_sync)
        except Exception:
            logger.exception("Failed to load models or FAISS index.")
            raise


async def _async_load_pdf_extractor_model():
    global pdf_llm
    if not _use_llama_cpp_backend() or not _is_pdf_extractor_configured():
        return
    if pdf_llm is not None:
        return

    async with _pdf_model_lock:
        if pdf_llm is not None:
            return
        try:
            await asyncio.to_thread(_load_pdf_extractor_model_sync)
        except Exception:
            logger.exception("Failed to load dedicated PDF extractor model.")
            raise

def load_models_if_needed():
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and not loop.is_closed():
        # Running inside an event loop; schedule a task if not already loaded.
        asyncio.create_task(_async_preload_models(preload_pdf=True))
    else:
        # No running loop (e.g., called from sync startup). Run the async loader.
        asyncio.run(_async_preload_models(preload_pdf=True))

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
            if response in _chunks_cache or query in _chunks_cache:
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

    if not _use_llama_cpp_backend():
        try:
            async with _model_lock:
                response = await asyncio.to_thread(
                    _ollama_chat_sync,
                    model=_ollama_chat_model(),
                    messages=[
                        {"role": "system", "content": PROMPT_SYSTEM_GUIDELINES},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=max_tokens,
                    stream=False,
                )
            return _ollama_response_text(response)
        except Exception:
            logger.exception("Ollama chat model call failed.")
            raise

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


async def _call_pdf_extractor_async(prompt_text: str, max_tokens: int = 380) -> str:
    await _async_load_models_and_index()  # ensure loaded

    if not _use_llama_cpp_backend():
        try:
            async with _pdf_model_lock:
                response = await asyncio.to_thread(
                    _ollama_chat_sync,
                    model=_ollama_extractor_model(),
                    messages=[
                        {"role": "system", "content": "You extract structured fields from technical documents. Return JSON only."},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=max_tokens,
                    stream=False,
                )
            return _ollama_response_text(response)
        except Exception:
            logger.exception("Ollama extractor model call failed.")
            raise

    await _async_load_pdf_extractor_model()
    model = pdf_llm or llm
    lock = _pdf_model_lock if model is pdf_llm else _model_lock

    if model is None:
        raise RuntimeError("No model available for PDF extraction.")

    async with lock:
        try:
            def _run():
                return model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You extract structured fields from technical documents. Return JSON only."},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=max_tokens,
                )

            result = await asyncio.to_thread(_run)
            text = result["choices"][0]["message"]["content"].strip()
            return text

        except Exception:
            logger.exception("PDF extractor model call failed.")
            raise


def _extract_stream_token(chunk: Dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""

    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    choice = choices[0] if isinstance(choices[0], dict) else {}

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    text = choice.get("text")
    if isinstance(text, str):
        return text

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    return ""


async def generate_answer_stream_async(query: str):
    if not query or not query.strip():
        return

    try:
        results = await get_similar_chunks(query, top_k=5)
        context_text = "\n\n".join([c for c, _ in results]) if results else ""
        if len(context_text) > 4000:
            context_text = context_text[:4000]

        prompt_text = _build_prompt(context_text, query)
        await _async_load_models_and_index()

        if not _use_llama_cpp_backend():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Tuple[str, Optional[str]]] = asyncio.Queue(maxsize=64)

            def _producer():
                response = None
                try:
                    response = _ollama_chat_sync(
                        model=_ollama_chat_model(),
                        messages=[
                            {"role": "system", "content": PROMPT_SYSTEM_GUIDELINES},
                            {"role": "user", "content": prompt_text},
                        ],
                        max_tokens=512,
                        stream=True,
                    )
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = json.loads(line)
                        message = data.get("message") if isinstance(data, dict) else None
                        token = message.get("content") if isinstance(message, dict) else ""
                        if token:
                            cleaned = token.replace("<think>", "").replace("</think>", "")
                            if cleaned:
                                asyncio.run_coroutine_threadsafe(queue.put(("token", cleaned)), loop).result()
                        if data.get("done"):
                            break
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop).result()
                finally:
                    if response is not None:
                        response.close()
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

            async with _model_lock:
                producer_task = asyncio.create_task(asyncio.to_thread(_producer))

                try:
                    while True:
                        event, payload = await queue.get()
                        if event == "token" and payload is not None:
                            yield payload
                            await asyncio.sleep(0)
                            continue
                        if event == "error":
                            raise RuntimeError(payload or "Ollama stream failed")
                        if event == "done":
                            break
                finally:
                    await producer_task
            return

        async with _model_lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Tuple[str, Optional[str]]] = asyncio.Queue(maxsize=64)

            def _producer():
                try:
                    stream_iter = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": PROMPT_SYSTEM_GUIDELINES},
                            {"role": "user", "content": prompt_text},
                        ],
                        max_tokens=512,
                        stream=True,
                    )

                    for chunk in stream_iter:
                        token = _extract_stream_token(chunk)
                        if not token:
                            continue

                        cleaned = token.replace("<think>", "").replace("</think>", "")
                        if cleaned:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("token", cleaned)),
                                loop,
                            ).result()
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("error", str(exc))),
                        loop,
                    ).result()
                finally:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("done", None)),
                        loop,
                    ).result()

            producer_task = asyncio.create_task(asyncio.to_thread(_producer))

            try:
                while True:
                    event, payload = await queue.get()
                    if event == "token" and payload is not None:
                        yield payload
                        await asyncio.sleep(0)
                        continue

                    if event == "error":
                        raise RuntimeError(payload or "Model stream failed")

                    if event == "done":
                        break
            finally:
                await producer_task
    except Exception:
        logger.exception("Streaming generation failed.")
        raise


def sanitize_output(text: str) -> str:
    if not text:
        return ""

    # Remove forbidden tags
    forbidden_tokens = ["<think>", "</think>"]
    for token in forbidden_tokens:
        text = text.replace(token, "")

    # Remove duplicated answers (simple heuristic)
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2 and parts[0] == parts[1]:
        text = parts[0]

    # Final cleanup
    return text.strip()

async def generate_answer_async(query: str) -> str:
    if not query or not query.strip():
        return ""
    try:
        results = await get_similar_chunks(query, top_k=5)
        context_text = "\n\n".join([c for c, _ in results]) if results else ""
        if len(context_text) > 4000:
            context_text = context_text[:4000]
        prompt_text = _build_prompt(context_text, query)
        raw = await _call_model_async(prompt_text, max_tokens=512)
        raw = sanitize_output(raw)

        try:
            asyncio.create_task(add_chat_to_faiss(query, raw))
        except Exception:
            logger.exception("Failed to schedule FAISS add task.")

        return raw.strip()
    except Exception:
        logger.exception("Generation failed.")
        # FIX
        return "Sorry, something went wrong during generation."


def generate_answer(query: str) -> List[str]:
    return asyncio.get_event_loop().run_until_complete(generate_answer_async(query))
