import logging
import os
from functools import lru_cache
from typing import Literal

logger = logging.getLogger("swarai.device")

DeviceName = Literal["cuda", "mps", "cpu"]


def _env_device_override() -> str:
    return os.getenv("SWARAI_DEVICE", os.getenv("DEVICE", "auto")).strip().lower()


@lru_cache(maxsize=1)
def resolve_device() -> DeviceName:
    """Return the best available runtime device, falling back to CPU safely."""
    override = _env_device_override()
    if override in {"cpu", "cuda", "mps"}:
        logger.info("Using configured device override: %s", override)
        return override  # type: ignore[return-value]

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s", gpu_name)
            return "cuda"

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            logger.info("Apple MPS GPU detected.")
            return "mps"
    except Exception:
        logger.exception("GPU detection failed; falling back to CPU.")

    logger.info("No GPU detected. Using CPU.")
    return "cpu"


def sentence_transformer_device() -> str:
    return resolve_device()


def default_llama_gpu_layers() -> int:
    """Use all GPU layers when CUDA exists; llama.cpp uses 0 for CPU."""
    return -1 if resolve_device() == "cuda" else 0


def is_cuda_available() -> bool:
    return resolve_device() == "cuda"

