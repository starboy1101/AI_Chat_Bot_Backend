import logging
from typing import Any

logger = logging.getLogger("swarai.llm_manager")

try:
    # import from package top-level backend module (the large one)
    from backend import generate_answer_async as legacy_generate_answer_async
    from backend import get_similar_chunks, add_chat_to_faiss
except Exception:
    try:
        from .backend import generate_answer_async as legacy_generate_answer_async
        from .backend import get_similar_chunks, add_chat_to_faiss
    except Exception:
        legacy_generate_answer_async = None
        get_similar_chunks = None
        add_chat_to_faiss = None
        logger.warning("legacy backend functions not found; implement generate_answer_async in backend.py or update import paths")

async def generate_answer_async(prompt: str) -> Any:
    clean_prompt = (
        "You are a concise, professional assistant. Provide ONLY a single, concise answer "
        "to the user's question. Do NOT generate multiple questions or follow-ups. "
        "Do NOT include 'User:' or 'Answer:' labels. Do NOT produce a list of Q&A pairs. "
        "Answer directly and briefly.\n\n"
        f"{prompt}"
    )

    if legacy_generate_answer_async:
        try:
            result = await legacy_generate_answer_async(clean_prompt)

            if isinstance(result, list) and result:
                return result[0]
            return result if isinstance(result, str) else str(result)
        except Exception:
            logger.exception("legacy_generate_answer_async failed in llm_manager wrapper.")

    logger.warning("Falling back to stub LLM responder")
    return f"[stub reply] {prompt}"



