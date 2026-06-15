from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from srs_generator.models import SRSProject
from srs_generator.utils import normalize_space
from srs_generator.validator import FALLBACK_TEXT, SRSValidator, fill_missing_srs_fields


LOCAL_AI_SRS_MODEL = "qwen2.5:7b"
DEFAULT_KEEP_ALIVE = "10m"
DEFAULT_JSON_RETRIES = 2
SCHEMA_PLACEHOLDER_VALUES = {
    "string",
    "short source quote or summary",
    "functional|non-functional|safety|cybersecurity",
    "urgent|high|medium|low",
}


class LocalAIUnavailableError(RuntimeError):
    """Raised when the optional local model backend is not reachable/configured."""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", (text or "").lower())


def _split_source_chunks(text: str, max_chars: int = 1400) -> List[str]:
    paragraphs = [normalize_space(part) for part in re.split(r"\n\s*\n+", text or "") if normalize_space(part)]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    if chunks:
        return chunks

    compact = normalize_space(text)
    return [compact[i : i + max_chars] for i in range(0, len(compact), max_chars) if compact[i : i + max_chars]]


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _env_int(*names: str, default: int) -> int:
    raw = _env_first(*names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(*names: str, default: Optional[int] = None) -> Optional[int]:
    raw = _env_first(*names)
    if raw is None:
        return default
    if raw.lower() in {"none", "null", "auto", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    raw = _env_first(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(*names: str, default: bool) -> bool:
    raw = _env_first(*names)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _is_schema_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", " ", value.strip().strip(".")).lower()
    return normalized in SCHEMA_PLACEHOLDER_VALUES or bool(re.fullmatch(r"<[^>]+>", value.strip()))


def _remove_schema_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _remove_schema_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_remove_schema_placeholders(item) for item in value]
    if _is_schema_placeholder(value):
        return ""
    return value


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("_")
    return cleaned[:80] or "uploaded_document"


def _is_empty_ai_scalar(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().strip(".")).lower()
    return text in {"", "none", "null", "n/a", "na", "not applicable", "not specified", "no", "no open questions"}


def _coerce_text_list(value: Any) -> List[str]:
    if value is None or _is_schema_placeholder(value) or _is_empty_ai_scalar(value):
        return []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if item is None or _is_schema_placeholder(item) or _is_empty_ai_scalar(item):
                continue
            if isinstance(item, dict):
                text = json.dumps(item, ensure_ascii=False)
            else:
                text = str(item).strip()
            if text:
                items.append(text)
        return items
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]

    text = str(value).strip()
    if not text:
        return []
    parts = [
        part.strip(" -\t\r\n")
        for part in re.split(r"(?:\n+|;\s+|(?:^|\n)\s*[-*]\s+)", text)
        if part.strip(" -\t\r\n")
    ]
    return parts or [text]


def _coerce_confidence(value: Any, default: float = 0.82) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1.0 and number <= 100.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    if not text or _is_schema_placeholder(value):
        return default
    priority_confidence = {
        "urgent": 0.9,
        "critical": 0.9,
        "high": 0.85,
        "medium": 0.72,
        "moderate": 0.72,
        "low": 0.55,
    }
    if text in priority_confidence:
        return priority_confidence[text]
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent_match:
        return max(0.0, min(1.0, float(percent_match.group(1)) / 100.0))
    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        number = float(number_match.group(0))
        if number > 1.0 and number <= 100.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))
    return default


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None or _is_schema_placeholder(value):
        return None
    if isinstance(value, bool):
        return value
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    if text in {"true", "yes", "y", "1", "critical", "high", "required", "feasible"}:
        return True
    if text in {"false", "no", "n", "0", "not critical", "not feasible"}:
        return False
    return None


def _normalize_requirement_type(value: Any) -> str:
    text = re.sub(r"[\s_]+", "-", str(value or "").strip().lower())
    if not text or _is_schema_placeholder(value):
        return "functional"
    if text in {"nonfunctional", "non-functional", "non-functional-requirement"}:
        return "non-functional"
    if "safety" in text:
        return "safety"
    if "cyber" in text or "security" in text:
        return "cybersecurity"
    if "non" in text and "functional" in text:
        return "non-functional"
    return "functional"


def _normalize_ai_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)

    confidence_value = metadata.get("confidence")
    if isinstance(confidence_value, str):
        confidence_text = confidence_value.strip().lower()
        if confidence_text in {"urgent", "critical", "high", "medium", "moderate", "low"} and not metadata.get("priority"):
            metadata["priority"] = confidence_value.strip().title()
    metadata["confidence"] = _coerce_confidence(confidence_value)

    if not metadata.get("priority") or _is_schema_placeholder(metadata.get("priority")):
        confidence = float(metadata["confidence"])
        metadata["priority"] = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.65 else "Low"

    for list_field in ("can_ids", "sds_references"):
        metadata[list_field] = _coerce_text_list(metadata.get(list_field))

    for text_field in ("customer_req_id", "milestone", "asil", "cal", "source_section", "extraction_method", "evidence"):
        value = metadata.get(text_field)
        if value is None or _is_schema_placeholder(value):
            metadata[text_field] = ""
        elif isinstance(value, (list, dict)):
            metadata[text_field] = json.dumps(value, ensure_ascii=False)
        else:
            metadata[text_field] = str(value).strip()

    return metadata


def _normalize_ai_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload["assumptions"] = _coerce_text_list(payload.get("assumptions"))
    payload["open_questions"] = _coerce_text_list(payload.get("open_questions"))

    for list_field in ("safety_requirements", "cybersecurity_requirements", "revision_history", "definitions", "references"):
        if not isinstance(payload.get(list_field), list):
            payload[list_field] = []
    if not isinstance(payload.get("interfaces"), dict):
        payload["interfaces"] = {}
    if "confidence" in payload:
        payload["confidence"] = _coerce_confidence(payload.get("confidence"), default=0.0)

    requirements = payload.get("requirements")
    if isinstance(requirements, dict):
        requirements = list(requirements.values())
    elif isinstance(requirements, str):
        requirements = [{"purpose": requirements}]
    elif not isinstance(requirements, list):
        requirements = []

    normalized_requirements: List[Dict[str, Any]] = []
    for index, requirement in enumerate(requirements, start=1):
        if isinstance(requirement, dict):
            item = dict(requirement)
        else:
            item = {"purpose": str(requirement).strip()}

        if not str(item.get("req_id") or "").strip() or _is_schema_placeholder(item.get("req_id")):
            item["req_id"] = f"AUTO_REQ_{index:03d}"
        if not str(item.get("logical_block") or "").strip() or _is_schema_placeholder(item.get("logical_block")):
            item["logical_block"] = "General"
        item["requirement_type"] = _normalize_requirement_type(item.get("requirement_type"))
        item["critical"] = _coerce_optional_bool(item.get("critical"))
        item["feasible"] = _coerce_optional_bool(item.get("feasible"))
        item["attachments"] = item.get("attachments") if isinstance(item.get("attachments"), list) else []
        item["field_evidence"] = item.get("field_evidence") if isinstance(item.get("field_evidence"), dict) else {}
        item["metadata"] = _normalize_ai_metadata(item.get("metadata"))
        normalized_requirements.append(item)

    payload["requirements"] = normalized_requirements
    return payload


def _requirement_chunk_score(chunk: str) -> int:
    lowered = (chunk or "").lower()
    score = 0
    if re.search(r"\b(?:req(?:uirement)?\s*id|swrs|srs[_-]?\w+|rq\d+|auto[-_]?req)\b", lowered):
        score += 4
    for cue in (
        "purpose",
        "input",
        "output",
        "process",
        "validation",
        "verification",
        "acceptance criteria",
        "derived requirement",
        "mandatory fields",
        "pre-loaded",
        "default value",
        "valid range",
        "data latency",
        "data retention",
        "data rate",
        "external events",
        "temporal events",
        "constraints",
        "testability",
        "critical",
        "feasible",
    ):
        if cue in lowered:
            score += 1
    if re.search(r"\b(?:shall|must|should)\b", lowered):
        score += 2
    return score


def _dedupe_chunks(chunks: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for chunk in chunks:
        key = re.sub(r"\W+", "", (chunk or "").lower())[:500]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


class BM25Retriever:
    def __init__(self, chunks: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = [chunk for chunk in chunks if normalize_space(chunk)]
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize(chunk) for chunk in self.chunks]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freq: Dict[str, int] = {}
        for doc in self.documents:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.chunks or not query:
            return []
        scores = [(self.chunks[index], self._score(_tokenize(query), index)) for index in range(len(self.chunks))]
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(chunk, score) for chunk, score in scores[:top_k] if score > 0]

    def _score(self, query_tokens: Iterable[str], doc_index: int) -> float:
        doc = self.documents[doc_index]
        if not doc:
            return 0.0
        frequencies: Dict[str, int] = {}
        for token in doc:
            frequencies[token] = frequencies.get(token, 0) + 1

        score = 0.0
        total_docs = len(self.documents)
        doc_len = self.doc_lengths[doc_index]
        for token in set(query_tokens):
            freq = frequencies.get(token, 0)
            if not freq:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
            score += idf * ((freq * (self.k1 + 1)) / denom)
        return score


@dataclass(frozen=True)
class LocalAIConfig:
    base_url: str
    text_model: str
    helper_model: str
    timeout_seconds: int
    max_context_chars: int
    num_ctx: int
    num_predict: Optional[int]
    retrieval_top_k: int
    baseline_max_chars: int
    temperature: float
    keep_alive: str
    num_gpu: int
    main_gpu: int
    stream: bool

    @classmethod
    def from_env(cls) -> "LocalAIConfig":
        text_model = _env_first(
            "LOCAL_AI_SRS_VISION_MODEL_2",
            "LOCAL_AI_SRS_TEXT_MODEL",
            "LOCAL_AI_SRS_MODEL",
            default=LOCAL_AI_SRS_MODEL,
        ) or LOCAL_AI_SRS_MODEL
        helper_model = _env_first("LOCAL_AI_SRS_HELPER_MODEL", default=text_model) or text_model
        return cls(
            base_url=(_env_first("LOCAL_AI_SRS_BASE_URL", "OLLAMA_BASE_URL", default="http://localhost:11434") or "").rstrip("/"),
            text_model=text_model,
            helper_model=helper_model,
            timeout_seconds=_env_int("LOCAL_AI_SRS_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS", default=1800),
            max_context_chars=_env_int(
                "LOCAL_AI_SRS_MAX_CONTEXT_CHARS",
                "LOCAL_AI_SRS_MAX_PROMPT_DOCUMENT_CHARS",
                "MAX_PROMPT_DOCUMENT_CHARS",
                default=24000,
            ),
            num_ctx=_env_int("LOCAL_AI_SRS_NUM_CTX", "OLLAMA_NUM_CTX", default=8192),
            num_predict=_env_optional_int("LOCAL_AI_SRS_NUM_PREDICT", "OLLAMA_NUM_PREDICT", default=None),
            retrieval_top_k=_env_int("LOCAL_AI_SRS_RETRIEVAL_TOP_K", default=12),
            baseline_max_chars=_env_int("LOCAL_AI_SRS_BASELINE_MAX_CHARS", default=8000),
            temperature=_env_float("LOCAL_AI_SRS_TEMPERATURE", "OLLAMA_TEMPERATURE", default=0.2),
            keep_alive=_env_first("LOCAL_AI_SRS_KEEP_ALIVE", "OLLAMA_KEEP_ALIVE", default=DEFAULT_KEEP_ALIVE) or DEFAULT_KEEP_ALIVE,
            num_gpu=_env_int("LOCAL_AI_SRS_NUM_GPU", "OLLAMA_NUM_GPU", default=-1),
            main_gpu=_env_int("LOCAL_AI_SRS_MAIN_GPU", "OLLAMA_MAIN_GPU", default=0),
            stream=_env_bool("LOCAL_AI_SRS_STREAM", "OLLAMA_STREAM", default=False),
        )


@dataclass(frozen=True)
class ColabSRSConfig:
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: int
    max_context_chars: int
    retrieval_top_k: int
    baseline_max_chars: int
    temperature: float
    num_ctx: int
    num_predict: Optional[int]
    api_key: Optional[str]

    @classmethod
    def from_env(cls) -> Optional["ColabSRSConfig"]:
        base_url = _env_first("COLAB_SRS_BASE_URL", "GOOGLE_COLAB_SRS_BASE_URL")
        if not base_url:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            endpoint=_env_first("COLAB_SRS_ENDPOINT", default="/generate-srs") or "/generate-srs",
            model=_env_first("COLAB_SRS_MODEL", default=LOCAL_AI_SRS_MODEL) or LOCAL_AI_SRS_MODEL,
            timeout_seconds=_env_int("COLAB_SRS_TIMEOUT_SECONDS", "LOCAL_AI_SRS_TIMEOUT_SECONDS", default=1800),
            max_context_chars=_env_int(
                "COLAB_SRS_MAX_CONTEXT_CHARS",
                "LOCAL_AI_SRS_MAX_CONTEXT_CHARS",
                "LOCAL_AI_SRS_MAX_PROMPT_DOCUMENT_CHARS",
                "MAX_PROMPT_DOCUMENT_CHARS",
                default=24000,
            ),
            retrieval_top_k=_env_int("COLAB_SRS_RETRIEVAL_TOP_K", "LOCAL_AI_SRS_RETRIEVAL_TOP_K", default=12),
            baseline_max_chars=_env_int("COLAB_SRS_BASELINE_MAX_CHARS", "LOCAL_AI_SRS_BASELINE_MAX_CHARS", default=8000),
            temperature=_env_float("COLAB_SRS_TEMPERATURE", "LOCAL_AI_SRS_TEMPERATURE", default=0.2),
            num_ctx=_env_int("COLAB_SRS_NUM_CTX", "LOCAL_AI_SRS_NUM_CTX", default=8192),
            num_predict=_env_optional_int("COLAB_SRS_NUM_PREDICT", "LOCAL_AI_SRS_NUM_PREDICT", default=None),
            api_key=_env_first("COLAB_SRS_API_KEY"),
        )


class ColabSRSClient:
    """HTTP client for the Colab FastAPI/ngrok SRS inference service."""

    def __init__(self, config: Optional[ColabSRSConfig] = None) -> None:
        resolved = config or ColabSRSConfig.from_env()
        if resolved is None:
            raise LocalAIUnavailableError("COLAB_SRS_BASE_URL is not configured.")
        self.config = resolved

    def generate_text(self, prompt: str) -> str:
        endpoint = self.config.endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "format": "json",
            "temperature": self.config.temperature,
            "num_ctx": self.config.num_ctx,
        }
        if self.config.num_predict is not None:
            payload["num_predict"] = self.config.num_predict

        headers = {"ngrok-skip-browser-warning": "true"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            response = requests.post(
                f"{self.config.base_url}{endpoint}",
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise LocalAIUnavailableError(
                f"Colab SRS API timed out after {self.config.timeout_seconds}s at {self.config.base_url}{endpoint}."
            ) from exc
        except requests.ConnectionError as exc:
            raise LocalAIUnavailableError(
                f"Colab SRS API is not reachable at {self.config.base_url}{endpoint}. "
                "Start the Colab FastAPI server and update COLAB_SRS_BASE_URL with the current ngrok URL."
            ) from exc
        except requests.RequestException as exc:
            raise LocalAIUnavailableError(f"Colab SRS API request failed at {self.config.base_url}{endpoint}: {exc}") from exc

        if response.status_code >= 400:
            text = response.text[:500]
            if response.status_code in {502, 503, 504} and "ngrok" in text.lower():
                raise LocalAIUnavailableError(
                    f"Colab/ngrok tunnel is unavailable at {self.config.base_url}{endpoint} "
                    f"(HTTP {response.status_code}). Restart the Colab API cell and update COLAB_SRS_BASE_URL if the ngrok URL changed."
                )
            raise LocalAIUnavailableError(f"Colab SRS API returned HTTP {response.status_code}: {text}")

        generated = self._extract_generated_text(response)
        if not generated:
            raise LocalAIUnavailableError("Colab SRS API returned an empty response.")
        return generated

    def _extract_generated_text(self, response: requests.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return response.text.strip()

        data = response.json()
        if isinstance(data, str):
            return data.strip()
        if not isinstance(data, dict):
            return ""

        for key in ("text", "response", "generated_text", "content", "output", "result"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = self._extract_from_mapping(value)
                if nested:
                    return nested

        message = data.get("message")
        if isinstance(message, dict):
            nested = self._extract_from_mapping(message)
            if nested:
                return nested

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                nested = self._extract_from_mapping(first)
                if nested:
                    return nested

        return ""

    def _extract_from_mapping(self, data: Dict[str, Any]) -> str:
        for key in ("content", "text", "response", "generated_text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = self._extract_from_mapping(value)
                if nested:
                    return nested
        return ""


class OllamaLocalSRSClient:
    def __init__(self, config: Optional[LocalAIConfig] = None) -> None:
        self.config = config or LocalAIConfig.from_env()

    def generate_text(self, prompt: str) -> str:
        options: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.num_ctx,
            "num_gpu": self.config.num_gpu,
            "main_gpu": self.config.main_gpu,
        }
        if self.config.num_predict is not None:
            options["num_predict"] = self.config.num_predict
        payload = {
            "model": self.config.text_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate complete SRS JSON. Return only valid JSON and never copy template placeholders.",
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": self.config.stream,
            "keep_alive": self.config.keep_alive,
            "options": options,
        }
        return self._post_generate(payload)

    def _post_generate(self, payload: Dict[str, Any]) -> str:
        try:
            endpoint = "/api/chat" if "messages" in payload else "/api/generate"
            response = requests.post(
                f"{self.config.base_url}{endpoint}",
                json=payload,
                timeout=self.config.timeout_seconds,
                stream=bool(payload.get("stream")),
            )
            
        except requests.Timeout as exc:
            model = payload.get("model", "configured model")
            raise LocalAIUnavailableError(
                f"Local Ollama model '{model}' timed out after {self.config.timeout_seconds}s at {self.config.base_url}. "
                "Reduce LOCAL_AI_SRS_MAX_CONTEXT_CHARS/LOCAL_AI_SRS_NUM_CTX or increase LOCAL_AI_SRS_TIMEOUT_SECONDS."
            ) from exc
        except requests.ConnectionError as exc:
            raise LocalAIUnavailableError(
                f"Local Ollama server is not reachable at {self.config.base_url}. "
                "Start Ollama and make sure the configured Qwen models are pulled."
            ) from exc
        except requests.RequestException as exc:
            raise LocalAIUnavailableError(f"Ollama request failed at {self.config.base_url}: {exc}") from exc

        if response.status_code == 404:
            model = payload.get("model", "configured model")
            raise LocalAIUnavailableError(f"Local model '{model}' is not installed or not available in Ollama.")
        if response.status_code >= 400:
            raise LocalAIUnavailableError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")

        if payload.get("stream"):
            chunks: List[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message") if isinstance(data, dict) else None
                if isinstance(message, dict):
                    chunks.append(str(message.get("content") or ""))
                else:
                    chunks.append(str(data.get("response") or ""))
                if data.get("done"):
                    break
            generated = "".join(chunks).strip()
        else:
            data = response.json()
            message = data.get("message") if isinstance(data, dict) else None
            if isinstance(message, dict):
                generated = str(message.get("content") or "").strip()
            else:
                generated = str(data.get("response") or "").strip()
        if not generated:
            raise LocalAIUnavailableError("Ollama returned an empty response.")
        return generated


class LocalAISRSGenerator:
    RETRIEVAL_QUERY = (
        "software requirements functional requirements non functional requirements system overview scope "
        "interfaces assumptions constraints validation acceptance criteria performance safety cybersecurity "
        "inputs outputs process workflow priority milestone"
    )

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client or self._default_client()

    def _default_client(self) -> Any:
        colab_config = ColabSRSConfig.from_env()
        if colab_config is not None:
            return ColabSRSClient(colab_config)
        return OllamaLocalSRSClient()

    def generate(
        self,
        *,
        source_text: str,
        baseline_project: Optional[SRSProject] = None,
        source_name: str = "uploaded_document",
    ) -> SRSProject:
        source_text = normalize_space(source_text)
        if not source_text:
            raise ValueError("No source text is available for local AI SRS generation.")

        context = self._build_bm25_context(source_text)
        prompt = self._build_generation_prompt(
            source_name=source_name,
            source_context=context,
            baseline_project=baseline_project,
        )
        try:
            payload = self._generate_json_payload_with_retries(
                prompt=prompt,
                source_name=source_name,
                source_context=context,
                baseline_project=baseline_project,
            )
        except LocalAIUnavailableError as exc:
            if baseline_project is None:
                raise
            return self._fallback_project_from_baseline(
                baseline_project=baseline_project,
                source_name=source_name,
                reason=str(exc),
            )
        if payload is None:
            if baseline_project is None:
                raise ValueError("Local AI response did not contain a JSON object and no baseline SRS is available.")
            return self._fallback_project_from_baseline(
                baseline_project=baseline_project,
                source_name=source_name,
                reason="Local AI response did not contain a JSON object after retries.",
            )
        payload = _normalize_ai_payload(_remove_schema_placeholders(payload))
        try:
            project = self._project_from_payload(payload)
        except Exception as exc:
            if baseline_project is None:
                raise
            self._save_invalid_ai_response(
                raw_response=json.dumps(payload, ensure_ascii=False, indent=2),
                source_name=source_name,
                attempt=0,
                error=f"AI payload validation failed after normalization: {exc}",
            )
            return self._fallback_project_from_baseline(
                baseline_project=baseline_project,
                source_name=source_name,
                reason=f"AI payload validation failed after normalization: {exc}",
            )
        if baseline_project is not None:
            self._merge_baseline_coverage(project, baseline_project)
        project.source_name = source_name
        self._finalize_project(project)
        project.validation_findings = []
        project = fill_missing_srs_fields(project)
        if baseline_project is not None:
            self._merge_baseline_coverage(project, baseline_project)
            project = fill_missing_srs_fields(project)
        self._finalize_project(project)
        model = getattr(self.client.config, "model", None) or getattr(self.client.config, "text_model", LOCAL_AI_SRS_MODEL)
        helper_model = getattr(self.client.config, "helper_model", model)
        backend_name = "colab_fastapi" if isinstance(self.client, ColabSRSClient) else "ollama"
        project.extraction_passes["local_ai_backend"] = backend_name
        project.extraction_passes["local_ai_model"] = model
        project.extraction_passes["local_ai_helper_model"] = helper_model
        project.extraction_passes["local_ai_retrieval"] = "bm25"
        project.extraction_passes["local_ai_num_ctx"] = self.client.config.num_ctx
        project.extraction_passes["local_ai_max_context_chars"] = self.client.config.max_context_chars
        return project

    def _generate_json_payload_with_retries(
        self,
        *,
        prompt: str,
        source_name: str,
        source_context: str,
        baseline_project: Optional[SRSProject],
    ) -> Optional[Dict[str, Any]]:
        attempts = max(1, _env_int("LOCAL_AI_SRS_JSON_RETRIES", "COLAB_SRS_JSON_RETRIES", default=DEFAULT_JSON_RETRIES) + 1)
        last_raw = ""
        last_error = ""

        for attempt in range(attempts):
            active_prompt = prompt if attempt == 0 else self._build_json_retry_prompt(
                source_name=source_name,
                source_context=source_context,
                baseline_project=baseline_project,
                previous_response=last_raw,
                previous_error=last_error,
            )
            try:
                raw = self.client.generate_text(active_prompt)
                last_raw = raw
                payload = self._extract_json_payload(raw)
                return payload
            except ValueError as exc:
                last_error = str(exc)
                self._save_invalid_ai_response(
                    raw_response=last_raw,
                    source_name=source_name,
                    attempt=attempt + 1,
                    error=last_error,
                )

        return None

    def _fallback_project_from_baseline(
        self,
        *,
        baseline_project: SRSProject,
        source_name: str,
        reason: str,
    ) -> SRSProject:
        project = self._copy_model(baseline_project)
        project.source_name = source_name
        project.validation_findings = []
        project = fill_missing_srs_fields(project)
        self._finalize_project(project)
        model = getattr(self.client.config, "model", None) or getattr(self.client.config, "text_model", LOCAL_AI_SRS_MODEL)
        backend_name = "colab_fastapi" if isinstance(self.client, ColabSRSClient) else "ollama"
        project.extraction_passes["local_ai_backend"] = backend_name
        project.extraction_passes["local_ai_model"] = model
        project.extraction_passes["local_ai_generation_status"] = "fallback_to_existing_srs_pipeline"
        project.extraction_passes["local_ai_generation_reason"] = reason
        project.extraction_passes["local_ai_retrieval"] = "baseline_existing_logic"
        return project

    def _build_bm25_context(self, source_text: str) -> str:
        chunks = _split_source_chunks(source_text)
        retriever = BM25Retriever(chunks)
        top_k = max(1, self.client.config.retrieval_top_k)
        ranked = retriever.search(self.RETRIEVAL_QUERY, top_k=top_k)
        ranked_chunks = [chunk for chunk, _score in ranked]
        requirement_chunks = [chunk for chunk in chunks if _requirement_chunk_score(chunk) > 0]
        selected = _dedupe_chunks([*(chunks[:1]), *requirement_chunks, *ranked_chunks, *chunks[:top_k]])

        context_parts: List[str] = []
        total_chars = 0
        max_chars = self.client.config.max_context_chars
        for index, chunk in enumerate(selected, start=1):
            piece = f"[Source excerpt {index}]\n{chunk}"
            if total_chars + len(piece) > max_chars:
                break
            context_parts.append(piece)
            total_chars += len(piece)
        return "\n\n".join(context_parts)

    def _build_generation_prompt(
        self,
        *,
        source_name: str,
        source_context: str,
        baseline_project: Optional[SRSProject],
    ) -> str:
        baseline = "{}"
        if baseline_project:
            baseline = json.dumps(self._compact_baseline_project(baseline_project), ensure_ascii=False)
        return f"""
You are an SRS generation engine. Generate a high-accuracy Software Requirements Specification JSON from the source excerpts.

Rules:
- Return only one valid JSON object. Do not use markdown fences.
- Use the source excerpts as the authority. Use the baseline JSON only as a hint.
- Treat any text inside angle brackets, such as <Purpose of the Functionality>, as a field description/instruction. Never copy angle brackets or placeholder text into the answer.
- Fill every SRS table field with a meaningful value. Use source facts first, then careful engineering inference from the requirement, subsystem, inputs, outputs, and domain.
- Do not output "{FALLBACK_TEXT}", "TBD", "N/A", "Not specified", "to be determined", or any unresolved placeholder.
- If a field is not explicitly present in the input, infer the safest useful value. If a field is truly not applicable, explain why in one concise sentence.
- For automotive requirements, REQ ID, purpose, validation/verification criteria, testability with testing phase, feasibility, criticality, and acceptance criteria are mandatory.
- Preserve requirement IDs from the source. If no ID exists, generate AUTO_REQ_001, AUTO_REQ_002, etc.
- Write every requirement as a clear, testable "shall" statement where possible.
- Keep fields concise but complete enough for a customer-facing SRS.

Output contract:
- Return a JSON object with these top-level keys: project_name, version, document_title, system_overview, scope, purpose, intended_audience, operating_environment, acceptance_criteria, assumptions, open_questions, requirements.
- requirements must be an array. Create one requirement object for every requirement boundary found in the source document: explicit IDs, table rows, "shall/must" statements, functional points, interface needs, timing constraints, data constraints, validation rules, safety/cybersecurity needs, and non-functional constraints.
- Include all requirement IDs found in the source. If the baseline JSON has requirement IDs, include those IDs unless the source clearly shows they are not real requirements.
- Each requirement object must include: logical_block, req_id, requirement_type, purpose, inputs, outputs, process, validation, acceptance_criteria, derived_requirement, access_restrictions, mandatory_fields, pre_loaded_values, default_values, valid_range_of_values, data_latency_period, data_retention_period, data_rate, external_events, temporal_events, constraints, effects_on_other_systems, assumptions, failure_scenario, action_on_failure, testability, status, critical, feasible, comments, metadata.
- metadata must include: customer_req_id, milestone, priority, asil, cal, can_ids, sds_references, source_section, extraction_method, evidence, confidence.
- Never output literal placeholder words like string, short source quote or summary, or enum lists such as Urgent|High|Medium|Low. Use a real inferred value for every field.
- If the source does not explicitly provide a field, infer it from the requirement purpose, inputs, outputs, subsystem, constraints, and validation context.

Source name:
{source_name}

Compact baseline JSON from existing generator:
{baseline}

BM25-selected source excerpts:
{source_context}
""".strip()

    def _build_json_retry_prompt(
        self,
        *,
        source_name: str,
        source_context: str,
        baseline_project: Optional[SRSProject],
        previous_response: str,
        previous_error: str,
    ) -> str:
        baseline = "{}"
        if baseline_project:
            baseline = json.dumps(self._compact_baseline_project(baseline_project), ensure_ascii=False)
        clipped_previous = normalize_space(previous_response)[-2500:]
        return f"""
Your previous SRS response was rejected because it was not parseable JSON.

Parser error:
{previous_error}

Return exactly one JSON object and nothing else. Do not include markdown, commentary, apologies, analysis, or code fences.

The JSON object must contain:
- project_name, version, document_title, system_overview, scope, purpose, intended_audience, operating_environment, acceptance_criteria, assumptions, open_questions, requirements
- requirements as an array with one item per requirement found in the source/baseline
- each requirement must include req_id, logical_block, requirement_type, purpose, inputs, outputs, process, validation, acceptance_criteria, constraints, failure_scenario, action_on_failure, testability, critical, feasible, metadata
- metadata must include customer_req_id, milestone, priority, evidence, confidence

Use real SRS values. Never output placeholder values such as string, TBD, N/A, Not specified, or enum examples.
If a field is not explicit in the source, infer it from the source context and baseline requirement.

Source name:
{source_name}

Baseline JSON from existing extractor:
{baseline}

Source excerpts:
{source_context}

Rejected previous response tail:
{clipped_previous}
""".strip()

    def _save_invalid_ai_response(self, *, raw_response: str, source_name: str, attempt: int, error: str) -> None:
        if not _env_bool("LOCAL_AI_SRS_SAVE_INVALID_RESPONSES", "COLAB_SRS_SAVE_INVALID_RESPONSES", default=True):
            return
        try:
            output_dir = Path(_env_first("LOCAL_AI_SRS_DEBUG_DIR", default="srs_generator/extracted_json/debug") or "srs_generator/extracted_json/debug")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"{_safe_filename_part(source_name)}_invalid_ai_attempt_{attempt}_{timestamp}.txt"
            debug_text = (
                f"error: {error}\n"
                f"source_name: {source_name}\n"
                f"attempt: {attempt}\n\n"
                f"{raw_response or ''}"
            )
            (output_dir / filename).write_text(debug_text, encoding="utf-8")
        except Exception:
            pass

    def _compact_baseline_project(self, project: SRSProject) -> Dict[str, Any]:
        payload = project.to_dict()
        compact_requirements: List[Dict[str, Any]] = []
        base_payload: Dict[str, Any] = {
            "project_name": payload.get("project_name"),
            "document_title": payload.get("document_title"),
            "system_overview": payload.get("system_overview"),
            "scope": payload.get("scope"),
            "requirements": compact_requirements,
        }
        max_chars = max(1000, self.client.config.baseline_max_chars)

        for requirement in payload.get("requirements", []):
            if not isinstance(requirement, dict):
                continue
            metadata = requirement.get("metadata") if isinstance(requirement.get("metadata"), dict) else {}
            compact_requirement = {
                "req_id": requirement.get("req_id"),
                "logical_block": requirement.get("logical_block"),
                "requirement_type": requirement.get("requirement_type"),
                "purpose": requirement.get("purpose"),
                "inputs": requirement.get("inputs"),
                "outputs": requirement.get("outputs"),
                "process": requirement.get("process"),
                "validation": requirement.get("validation"),
                "acceptance_criteria": requirement.get("acceptance_criteria"),
                "priority": metadata.get("priority"),
                "customer_req_id": metadata.get("customer_req_id"),
                "milestone": metadata.get("milestone"),
            }
            compact_requirements.append(compact_requirement)
            if len(json.dumps(base_payload, ensure_ascii=False)) > max_chars:
                compact_requirements.pop()
                break
        return base_payload

    def _merge_baseline_coverage(self, project: SRSProject, baseline_project: SRSProject) -> None:
        validator = SRSValidator()
        project_fields = (
            "project_name",
            "document_title",
            "system_overview",
            "scope",
            "purpose",
            "intended_audience",
            "operating_environment",
            "acceptance_criteria",
            "diagnostics",
        )
        for field in project_fields:
            current = getattr(project, field, None)
            baseline_value = getattr(baseline_project, field, None)
            if (
                hasattr(project, field)
                and not validator._is_missing_text_value(baseline_value)
                and (validator._is_missing_text_value(current) or _is_schema_placeholder(current))
            ):
                setattr(project, field, baseline_value)

        ai_by_id = {req.req_id.upper(): req for req in project.requirements if req.req_id}
        requirement_fields = (
            *validator.REQUIRED_REQUIREMENT_FIELDS,
            *validator.OPTIONAL_REQUIREMENT_FIELDS,
            "status",
            "comments",
        )
        metadata_fields = (
            "customer_req_id",
            "milestone",
            "priority",
            "asil",
            "cal",
            "source_section",
            "extraction_method",
            "evidence",
            "confidence",
        )

        for baseline_req in baseline_project.requirements:
            key = baseline_req.req_id.upper()
            ai_req = ai_by_id.get(key)
            if ai_req is None:
                project.requirements.append(self._copy_model(baseline_req))
                ai_by_id[key] = project.requirements[-1]
                continue

            for field in requirement_fields:
                if not hasattr(ai_req, field):
                    continue
                current = getattr(ai_req, field, None)
                baseline_value = getattr(baseline_req, field, None)
                if validator._is_missing_requirement_value(current) or _is_schema_placeholder(current):
                    if not validator._is_missing_requirement_value(baseline_value):
                        setattr(ai_req, field, baseline_value)

            for field in metadata_fields:
                current = getattr(ai_req.metadata, field, None)
                baseline_value = getattr(baseline_req.metadata, field, None)
                if validator._is_missing_text_value(current) or _is_schema_placeholder(current):
                    if not validator._is_missing_text_value(baseline_value):
                        setattr(ai_req.metadata, field, baseline_value)

            if not ai_req.metadata.can_ids and baseline_req.metadata.can_ids:
                ai_req.metadata.can_ids = list(baseline_req.metadata.can_ids)
            if not ai_req.metadata.sds_references and baseline_req.metadata.sds_references:
                ai_req.metadata.sds_references = list(baseline_req.metadata.sds_references)

    def _copy_model(self, value: Any) -> Any:
        if hasattr(value, "model_copy"):
            return value.model_copy(deep=True)
        return value.copy(deep=True)

    def _extract_json_payload(self, raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        candidates = self._extract_json_object_candidates(cleaned)
        if not candidates:
            raise ValueError("Local AI response did not contain a JSON object.")

        for data in reversed(candidates):
            if isinstance(data.get("requirements"), list):
                return data
        return candidates[-1]

    def _extract_json_object_candidates(self, cleaned: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        depth = 0
        in_string = False
        escaped = False
        start: Optional[int] = None
        for index, char in enumerate(cleaned):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        data = json.loads(cleaned[start : index + 1])
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(data, dict):
                        candidates.append(data)
                    start = None
        return candidates

    def _project_from_payload(self, payload: Dict[str, Any]) -> SRSProject:
        payload = _normalize_ai_payload(payload)
        for index, requirement in enumerate(payload["requirements"], start=1):
            if not isinstance(requirement, dict):
                continue
            if not str(requirement.get("req_id") or "").strip() or _is_schema_placeholder(requirement.get("req_id")):
                requirement["req_id"] = f"AUTO_REQ_{index:03d}"
            if not str(requirement.get("logical_block") or "").strip() or _is_schema_placeholder(requirement.get("logical_block")):
                requirement["logical_block"] = "General"
            if not str(requirement.get("requirement_type") or "").strip() or _is_schema_placeholder(requirement.get("requirement_type")):
                requirement["requirement_type"] = "functional"
            requirement.setdefault("metadata", {})
            if isinstance(requirement["metadata"], dict):
                requirement["metadata"].setdefault("extraction_method", "local_ai_bm25")
                requirement["metadata"].setdefault("confidence", 0.82)
        if hasattr(SRSProject, "model_validate"):
            return SRSProject.model_validate(payload)  # type: ignore[attr-defined]
        return SRSProject.parse_obj(payload)

    def _finalize_project(self, project: SRSProject) -> None:
        project.safety_requirements = [req for req in project.requirements if req.requirement_type == "safety"]
        project.cybersecurity_requirements = [
            req for req in project.requirements if req.requirement_type == "cybersecurity"
        ]
        if project.requirements and not project.confidence:
            scores = [float(req.metadata.confidence or 0.0) for req in project.requirements]
            project.confidence = round(sum(scores) / len(scores), 3)


def build_srs_comparison(original: SRSProject, ai_project: SRSProject) -> Dict[str, Any]:
    original_metrics = _project_metrics(original)
    ai_metrics = _project_metrics(ai_project)
    return {
        "original": original_metrics,
        "ai_generated": ai_metrics,
        "recommendation": _comparison_recommendation(original_metrics, ai_metrics),
    }


def _project_metrics(project: SRSProject) -> Dict[str, Any]:
    fallback_fields = project.extraction_passes.get("fallback_fields", [])
    findings = project.validation_findings or []
    local_ai_backend = project.extraction_passes.get("local_ai_backend")
    local_ai_model = project.extraction_passes.get("local_ai_model")
    local_ai_status = project.extraction_passes.get("local_ai_generation_status")
    return {
        "project_name": project.project_name,
        "requirements": len(project.requirements),
        "confidence": round(float(project.confidence or 0.0), 3),
        "local_ai_backend": local_ai_backend,
        "local_ai_model": local_ai_model,
        "local_ai_generation_status": local_ai_status,
        "fallback_fields": len(fallback_fields) if isinstance(fallback_fields, list) else 0,
        "validation_warnings": len([finding for finding in findings if getattr(finding, "severity", "") == "warning"]),
        "functional": len([req for req in project.requirements if req.requirement_type == "functional"]),
        "non_functional": len([req for req in project.requirements if req.requirement_type == "non-functional"]),
        "safety": len([req for req in project.requirements if req.requirement_type == "safety"]),
        "cybersecurity": len([req for req in project.requirements if req.requirement_type == "cybersecurity"]),
        "requirement_ids": [req.req_id for req in project.requirements[:25]],
    }


def _comparison_recommendation(original: Dict[str, Any], ai_project: Dict[str, Any]) -> str:
    original_score = (original["requirements"] * 2) + original["confidence"] - original["fallback_fields"]
    ai_score = (ai_project["requirements"] * 2) + ai_project["confidence"] - ai_project["fallback_fields"]
    if ai_score > original_score + 1:
        return "AI generated SRS has broader extracted coverage. Review it first, then confirm against the source."
    if original_score > ai_score + 1:
        return "Original SRS is more conservative and likely safer when source traceability matters most."
    return "Both versions are close. Prefer the one with clearer requirement wording after customer review."
