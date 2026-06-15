from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict


def normalize_space(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_inline(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def safe_filename(value: str, default: str = "generated_srs") -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("._")
    return name or default


def truthy_text(value: Any) -> bool | None:
    text = compact_inline(value).lower()
    if not text:
        return None
    if text in {"yes", "y", "true", "critical", "applicable", "required"}:
        return True
    if text in {"no", "n", "false", "not critical", "not applicable", "na", "n/a"}:
        return False
    if any(token in text for token in ("yes", "critical", "required")):
        return True
    if any(token in text for token in ("no", "not applicable", "not required")):
        return False
    return None


def read_simple_yaml(path: str | Path) -> Dict[str, Any]:
    """Tiny settings reader for key: value YAML files without requiring PyYAML."""
    settings: Dict[str, Any] = {}
    p = Path(path)
    if not p.exists():
        return settings
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        settings[key.strip()] = value
    return settings


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_logger(name: str) -> logging.Logger:
    try:
        import structlog

        return structlog.get_logger(name)  # type: ignore[return-value]
    except Exception:
        return logging.getLogger(name)

