import logging
from typing import Any, List

logger = logging.getLogger("swarai.faiss_manager")

try:
    from backend import get_similar_chunks, add_chat_to_faiss
except Exception:
    try:
        from .backend import get_similar_chunks, add_chat_to_faiss
    except Exception:
        get_similar_chunks = None
        add_chat_to_faiss = None
        logger.warning("FAISS helper functions not present in legacy backend")

async def search(query: str, top_k: int = 5) -> List[Any]:
    if get_similar_chunks:
        return await get_similar_chunks(query, top_k=top_k)
    return []

async def add(query: str, response: str) -> bool:
    if add_chat_to_faiss:
        return await add_chat_to_faiss(query, response)
    return False
