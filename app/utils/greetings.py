import difflib
import logging

from app.utils.common import normalize_text

logger = logging.getLogger("swarai.greetings")

GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
GREETING_REPLY = (
    "Hello! I'm SwarAI.\n\n"
    "I can help you understand and solve queries related to WASAPI, IAudioClient, APOs, "
    "audio processing, and other Windows audio architecture topics.\n\n"
    "Ask me anything related to audio!"
)


def is_greeting(text: str) -> bool:
    if not text:
        return False
    t = normalize_text(text)
    tokens = [tok for tok in t.split() if tok]
    if any(tok in GREETING_KEYWORDS for tok in tokens):
        return True
    for g in GREETING_KEYWORDS:
        if t == g or t.startswith(g + " "):
            return True
    best = difflib.get_close_matches(t, list(GREETING_KEYWORDS), n=1, cutoff=0.75)
    if best:
        return True
    for tok in tokens:
        best = difflib.get_close_matches(tok, list(GREETING_KEYWORDS), n=1, cutoff=0.8)
        if best:
            return True
    return False
