import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Tuple

from app.services.backend import _call_pdf_extractor_async, sanitize_output, split_text_safely

logger = logging.getLogger(__name__)
QUESTION_BATCH_SIZE = max(1, int(os.getenv("PDF_QUESTION_BATCH_SIZE", "14")))
PDF_EXTRACT_CHUNK_LEN = max(1200, int(os.getenv("PDF_EXTRACT_CHUNK_LEN", "3500")))
PDF_EXTRACT_MAX_TOKENS = max(180, int(os.getenv("PDF_EXTRACT_MAX_TOKENS", "380")))
PDF_EXTRACT_JSON_RETRIES = max(0, int(os.getenv("PDF_EXTRACT_JSON_RETRIES", "1")))

ALLOWED_QUESTION_IDS = {
    "service_select":"Type of service requested, Porting related requirement or target, Type of optimization requested",
    "Optimization_type": "Type of optimization requested, e.g. design level optimization, intrinsic optimization, algorithm optimization",
    "App_type": "Type of application or use case, e.g. audio porting, audio optimization, audio application development, DSP optimization",
    "Porting_question_1": "target of the audio algorithm running on currently and needs to be ported from",
    "Porting_question_2": "is there a requirement for cross compliance or multi-platform support",
    "DSP_Processor": "DSP processor type, e.g. Qualcomm Hexagon, HiFi 4, HiFi 5, TI C6000, QCOM 855",
    "Application": "Application type, e.g. arm, intel, amid etc.",
    "Audio_Interface": "Audio interface type, e.g. elite framework, audio reach freamework, Capi V1/V2 Audio Interface, etc.",
    "Audio_Params_1": "Pcm sample size, e.g. 16-bit, 24-bit, 32-bit",
    "Audio_Params_2": "sampling frequency, e.g. 48 kHz, 96 kHz, 192 kHz",
    "Audio_Params_3": "audio format, e.g. mono, stereo, 5.1, 7.1",
    "Audio_Tech_1": "audio processing modules, e.g. echo cancellation, noise suppression, automatic gain control, etc.",
    "Audio_Tech_2": "current platforms supported by the audio processing modules, e.g. Qualcomm Hexagon DSP, TI C6000 DSP, etc.",
    "CodeBase_1": "languages used in the code base, e.g. C, C++, Python, etc.",
    "CodeBase_2": "type of sample app provided, e.g. voice assistant, music player, etc.",
    "CodeBase_3": "code size in lines of code or approximate size, e.g. 10K LOC, 50K LOC, etc.",
    "CodeBase_4": "memory requirement of the code base, e.g. 100MB, 500MB, etc.",
    "CodeBase_5": "is the source code available, e.g. yes, no",
    "CodeBase_6": "code implementation type, e.g. reference implementation, production-ready implementation, etc.",
    "TargetPlatform_1": "target platform to port, e.g. Qualcomm platform, TI platform, Hexagon DSP, etc.",
    "TargetPlatform_2": "chipset details of the target platform, e.g. QCOM 845, QCOM 855, TI TDA2x, Snapdragon 865, etc.",
}

EVIDENCE_PATTERNS = {
    "Audio_Params_1": re.compile(r"\b(8|16|24|32)\s*[- ]?bit\b|\b(bit[- ]?depth|pcm sample size)\b", re.I),
    "Audio_Params_2": re.compile(r"\b\d+(\.\d+)?\s*(k?hz)\b|\b(sample rate|sampling frequency|sampling rate)\b", re.I),
    "Audio_Params_3": re.compile(r"\b(mono|stereo|multi channel|5\.1|7\.1|channel)\b", re.I),
}

STRICT_EVIDENCE_REQUIRED_QIDS = {"Audio_Params_1", "Audio_Params_2"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_answer(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        return incoming

    existing_conf = _to_float(existing.get("confidence"), 0.0)
    incoming_conf = _to_float(incoming.get("confidence"), 0.0)

    if incoming_conf > existing_conf:
        return incoming

    if (
        incoming_conf == existing_conf
        and isinstance(existing.get("value"), list)
        and isinstance(incoming.get("value"), list)
    ):
        merged = list(dict.fromkeys(existing["value"] + incoming["value"]))
        return {**existing, "value": merged}

    return existing


def _normalize_model_output(data: Any) -> Dict[str, Dict[str, Any]]:
    """
    Accept both possible model shapes:
    1) {"ProductA": {"question_id": {...}}}
    2) {"question_id": {...}}  -> treated as product="default"
    """
    if not isinstance(data, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    direct_questions: Dict[str, Any] = {}

    for key, value in data.items():
        if key in ALLOWED_QUESTION_IDS and isinstance(value, dict):
            direct_questions[key] = value
            continue

        if isinstance(value, dict):
            normalized[str(key)] = value

    if direct_questions:
        normalized.setdefault("default", {}).update(direct_questions)

    return normalized


def _iter_question_batches() -> Iterator[List[str]]:
    ordered = list(ALLOWED_QUESTION_IDS.keys())
    for i in range(0, len(ordered), QUESTION_BATCH_SIZE):
        yield ordered[i : i + QUESTION_BATCH_SIZE]


def _build_question_block(questions: Dict[str, Any], qids: List[str]) -> str:
    lines = []
    for qid in qids:
        semantic_description = ALLOWED_QUESTION_IDS[qid]
        q = questions.get(qid, {})
        text = q.get("text")
        options = _compact_options(q.get("options", []))
        lines.append(
            f'- "{qid}":\n'
            f'  ui_question: {text or "N/A"}\n'
            f'  semantic_description: {semantic_description}\n'
            f"  options_hint: {options}"
        )

    return "\n".join(lines)


def _compact_options(raw_options: Any, max_items: int = 5, max_label_len: int = 48) -> str:
    if not isinstance(raw_options, list) or not raw_options:
        return "[]"

    labels: List[str] = []
    for opt in raw_options:
        if len(labels) >= max_items:
            break
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).strip()
        else:
            label = str(opt).strip()
        if not label:
            continue
        labels.append(label[:max_label_len])

    if len(raw_options) > len(labels):
        labels.append("...")
    return json.dumps(labels, ensure_ascii=False)


def _is_evidence_relevant(qid: str, value: Any, evidence: str) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False

    evidence_text = evidence.strip().lower()
    value_text = str(value).strip().lower() if value is not None else ""

    # If the exact value text appears in evidence, consider it relevant.
    if value_text and value_text in evidence_text:
        return True

    pattern = EVIDENCE_PATTERNS.get(qid)
    if pattern is None:
        return True

    return bool(pattern.search(evidence_text))


def _contains_evidence_snippet(chunk: str, evidence: str) -> bool:
    if not isinstance(chunk, str) or not isinstance(evidence, str):
        return False
    if not chunk.strip() or not evidence.strip():
        return False

    chunk_n = re.sub(r"\s+", " ", chunk.strip().lower())
    evidence_n = re.sub(r"\s+", " ", evidence.strip().lower())
    return evidence_n in chunk_n


def _load_model_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    text = raw.strip().replace("```json", "").replace("```", "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # If model adds trailing text after a valid JSON object, decode prefix.
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Truncation repair: trim dangling tail and close open braces.
    start = text.find("{")
    if start < 0:
        return {}
    candidate = text[start:]

    max_trim = min(len(candidate), 1000)
    for trim in range(0, max_trim + 1):
        piece = candidate[:-trim] if trim else candidate
        piece = re.sub(r",\s*$", "", piece, flags=re.S)
        piece = re.sub(r',\s*"[^"]*$', "", piece, flags=re.S)
        piece = re.sub(r':\s*"[^"]*$', "", piece, flags=re.S)
        piece = re.sub(r":\s*$", "", piece, flags=re.S)

        if not piece.strip():
            continue

        opens = piece.count("{")
        closes = piece.count("}")
        if opens < closes:
            continue

        repaired = piece + ("}" * (opens - closes))
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    return {}


async def _extract_batch_json(prompt: str) -> Tuple[Dict[str, Any], str]:
    total_attempts = 1 + PDF_EXTRACT_JSON_RETRIES
    last_raw = ""

    for attempt in range(total_attempts):
        prompt_text = prompt
        if attempt > 0:
            prompt_text = (
                f"{prompt}\n\n"
                "IMPORTANT: Return only one valid JSON object. "
                "Do not include comments, markdown, or extra text."
            )
        raw = await _call_pdf_extractor_async(prompt_text, max_tokens=PDF_EXTRACT_MAX_TOKENS)
        raw = sanitize_output(raw)
        data = _load_model_json(raw)
        if data:
            return data, raw
        last_raw = raw

    return {}, last_raw


async def extract_answers_from_pdf(
    pdf_text: str,
    questions: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    chunks = split_text_safely(pdf_text, max_len=PDF_EXTRACT_CHUNK_LEN)
    total_chunks = len(chunks)
    logger.info("[DOC-EXTRACT] Starting fallback LLM extraction: %d chunks", total_chunks)
    question_batches = [
        (batch_qids, set(batch_qids), _build_question_block(questions, batch_qids))
        for batch_qids in _iter_question_batches()
    ]

    for idx, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        logger.info("[DOC-EXTRACT] Extracting chunk %d/%d", idx, total_chunks)

        for batch_idx, (batch_qids, batch_qid_set, question_block) in enumerate(question_batches, start=1):
            logger.debug(
                "[DOC-EXTRACT] Chunk %d/%d batch %d qids=%s",
                idx,
                total_chunks,
                batch_idx,
                batch_qids,
            )

            prompt = f"""
You are a senior audio systems engineer and requirements analyst.

Your task is to extract structured requirements from a technical description.
The document may be informal, descriptive, or narrative.
The document may describe multiple products.

Treat each product independently. Do not mix requirements.
Match each question_id using the semantic_description and examples provided for that question_id.
Do not rely only on literal keyword overlap.
Follow the same question_id order shown in the Questions section.
Only use the question_ids shown in the Questions section for this response.
Do not return any extra question_ids.
If a value is missing for a question_id, omit that question_id from output.

Return confidence between 0.0 and 1.0.
If ambiguous, lower confidence.

Include exact evidence text for each extracted value.
Evidence must support that specific question_id only.
Do not reuse the same evidence sentence for unrelated question_ids.
Do not assume values when they are not explicitly present in the text.
Especially do not assume PCM sample size or sampling frequency.

Return only valid JSON.
No markdown.
No explanations.

Format:
{{
  "<product_name>": {{
    "<question_id>": {{
      "value": string | [string] | null,
      "confidence": number,
      "evidence": string
    }}
  }}
}}

Questions:
{question_block}

Document chunk:
{chunk}

Output JSON only.
"""

            data, raw = await _extract_batch_json(prompt)

            if not data:
                logger.warning(
                    "[DOC-EXTRACT] Chunk %d/%d batch %d returned non-JSON/truncated output after %d attempts. raw=%s",
                    idx,
                    total_chunks,
                    batch_idx,
                    (1 + PDF_EXTRACT_JSON_RETRIES),
                    (raw[:300] if raw else ""),
                )
                continue

            product_blocks = _normalize_model_output(data)
            if not product_blocks:
                continue

            for product, answers in product_blocks.items():
                if not isinstance(answers, dict):
                    continue

                product_results = results.setdefault(product, {})

                for qid, obj in answers.items():
                    if qid not in batch_qid_set or qid not in questions:
                        continue
                    if not isinstance(obj, dict):
                        continue

                    value = obj.get("value")
                    confidence = _to_float(obj.get("confidence", 0.0))
                    evidence = obj.get("evidence", "")
                    evidence_ok = _is_evidence_relevant(qid, value, evidence) if evidence else False
                    value_text = str(value).strip().lower() if value is not None else ""
                    explicit_value_in_evidence = bool(evidence) and bool(value_text) and value_text in evidence.lower()
                    if evidence and not evidence_ok:
                        logger.debug(
                            "[DOC-EXTRACT] Dropping weak evidence product=%s qid=%s value=%r evidence=%s",
                            product,
                            qid,
                            value,
                            evidence[:200],
                        )
                        evidence = ""

                    if (
                        qid in STRICT_EVIDENCE_REQUIRED_QIDS
                        and (
                            not evidence
                            or not explicit_value_in_evidence
                            or not _contains_evidence_snippet(chunk, evidence)
                        )
                    ):
                        logger.debug(
                            "[DOC-EXTRACT] Dropping assumed value product=%s qid=%s value=%r due to missing/invalid explicit evidence",
                            product,
                            qid,
                            value,
                        )
                        value = None
                        confidence = 0.0

                    incoming = {
                        "value": value,
                        "confidence": confidence,
                        "source": "pdf",
                        "evidence": evidence,
                    }

                    product_results[qid] = _merge_answer(product_results.get(qid), incoming)

                    logger.debug(
                        "[DOC-EXTRACT] Extracted product=%s qid=%s value=%r confidence=%.3f evidence=%s",
                        product,
                        qid,
                        incoming["value"],
                        incoming["confidence"],
                        (incoming.get("evidence") or "")[:200],
                    )

    if not results:
        results = {"default": {}}

    final: Dict[str, Dict[str, Any]] = {}
    for product, product_answers in results.items():
        final[product] = {}
        for qid in ALLOWED_QUESTION_IDS:
            entry = product_answers.get(qid)
            if entry and entry.get("value") not in (None, "", []):
                final[product][qid] = entry
            else:
                final[product][qid] = {
                    "value": None,
                    "confidence": 0.0,
                    "source": "pdf",
                    "evidence": "",
                }

    for product, product_answers in final.items():
        for qid, entry in product_answers.items():
            if entry.get("value") not in (None, "", []):
                logger.debug(
                    "[DOC-EXTRACT] Final product=%s qid=%s value=%r confidence=%.3f",
                    product,
                    qid,
                    entry.get("value"),
                    _to_float(entry.get("confidence", 0.0)),
                )

    logger.info("[DOC-EXTRACT] Fallback LLM extraction completed")
    return final
