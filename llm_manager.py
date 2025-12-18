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
    clean_prompt = prompt

    if legacy_generate_answer_async:
        try:
            result = await legacy_generate_answer_async(clean_prompt)

            if isinstance(result, str):
                return result
            return str(result)
        except Exception:
            logger.exception("legacy_generate_answer_async failed in llm_manager wrapper.")

    logger.warning("Falling back to stub LLM responder")
    return "Sorry, I'm temporarily unable to generate a response."




