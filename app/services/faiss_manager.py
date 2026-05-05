import logging
from typing import Any, List

from app.services.backend import add_chat_to_faiss, get_similar_chunks

logger = logging.getLogger("swarai.faiss_manager")


async def search(query: str, top_k: int = 5) -> List[Any]:
    return await get_similar_chunks(query, top_k=top_k)


async def add(query: str, response: str) -> bool:
    return await add_chat_to_faiss(query, response)
