import logging
from typing import Any

from app.services.backend import (
    generate_answer_async as legacy_generate_answer_async,
    generate_answer_stream_async as legacy_generate_answer_stream_async,
)

logger = logging.getLogger("swarai.llm_manager")


async def generate_answer_async(prompt: str) -> Any:
    clean_prompt = prompt

    try:
        result = await legacy_generate_answer_async(clean_prompt)

        if isinstance(result, str):
            return result
        return str(result)
    except Exception:
        logger.exception("legacy_generate_answer_async failed in llm_manager wrapper.")
        logger.warning("Falling back to stub LLM responder")
        return "Sorry, I'm temporarily unable to generate a response."


async def generate_answer_stream_async(prompt: str):
    clean_prompt = prompt

    try:
        async for token in legacy_generate_answer_stream_async(clean_prompt):
            yield token
        return
    except Exception:
        logger.exception("legacy_generate_answer_stream_async failed in llm_manager wrapper.")

    logger.warning("Falling back to stub LLM stream responder")
    yield "Sorry, I'm temporarily unable to generate a response."
