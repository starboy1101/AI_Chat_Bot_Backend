# stdlib
import base64
import binascii
import html
import json
import re
import zipfile
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

# logging
import logging

from app.core.config import FLOW_FILE
from app.models.schemas import ChatRequest
from app.services.backend import generate_answer_async, matches_services_trigger
from app.services.chat_persistence import (
    persist_chat_pair,
    update_session_title_if_needed
)
from app.services.extract_pdf_text import extract_pdf_text, PDFParseError
from app.services.flows import conversation_flow, flow_manager
from app.services.pdf_generator import generate_final_requirements_pdf
from app.services.pdf_question_extractor import ALLOWED_QUESTION_IDS, extract_answers_from_pdf
from app.services.storage_utils import upload_file_to_supabase, upload_pdf_to_supabase
from app.state.chat_state import user_sessions
from app.utils.greetings import GREETING_REPLY

logger = logging.getLogger(__name__)

with open(FLOW_FILE, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

REQUIREMENT_IDS_IN_ORDER = list(ALLOWED_QUESTION_IDS.keys())

SERVICE_EXCLUDED_REQUIREMENTS = {
    "porting": {"Optimization_type", "App_type"},
    "optimization": {"Porting_type", "Porting_question_1", "Porting_question_2", "App_type", "TargetPlatform_1", "TargetPlatform_2"},
    "audio_app": {"Optimization_type", "Porting_type", "Porting_question_1", "Porting_question_2"},
}

PDF_SKIP_INPUTS = {
    "no",
    "n",
    "no pdf",
    "skip",
    "continue without pdf",
    "dont have pdf",
    "don't have pdf",
}

_DOC_MIME_TYPES = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_TEXT_MIME_TYPES = {"text/plain", "text"}


def _infer_attachment_kind(attachment: Any) -> Optional[str]:
    if not isinstance(attachment, dict):
        return None

    raw_type = str(attachment.get("type", "")).strip().lower()
    raw_name = str(attachment.get("name", "")).strip().lower()

    if raw_name.endswith(".pdf") or raw_type in {"pdf", "application/pdf"}:
        return "pdf"
    if raw_name.endswith(".docx") or raw_type in {"docx"} or raw_type in _DOC_MIME_TYPES:
        return "docx"
    if raw_name.endswith(".doc") or raw_type in {"doc"}:
        return "doc"
    if raw_name.endswith(".txt") or raw_type in {"txt", "plain", "text"} or raw_type in _TEXT_MIME_TYPES:
        return "txt"
    return None


def _normalize_extracted_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_docx_text(docx_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(docx_bytes)) as archive:
            names = [name for name in archive.namelist() if name.startswith("word/") and name.endswith(".xml")]
            text_parts: List[str] = []
            for name in names:
                xml = archive.read(name).decode("utf-8", errors="ignore")
                xml = re.sub(r"</w:p>", "\n", xml)
                xml = re.sub(r"</w:tr>", "\n", xml)
                xml = re.sub(r"<[^>]+>", " ", xml)
                plain = _normalize_extracted_text(html.unescape(xml))
                if plain:
                    text_parts.append(plain)
        merged = _normalize_extracted_text("\n\n".join(text_parts))
    except Exception as exc:
        raise ValueError(str(exc))

    if len(merged) < 40:
        raise ValueError("DOCX unreadable or empty")
    return merged


def _extract_txt_text(txt_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"):
        try:
            decoded = txt_bytes.decode(encoding, errors="strict")
            text = _normalize_extracted_text(decoded)
            if len(text) >= 40:
                return text
        except Exception:
            continue

    decoded = txt_bytes.decode("utf-8", errors="ignore")
    text = _normalize_extracted_text(decoded)
    if len(text) < 40:
        raise ValueError("TXT unreadable or empty")
    return text


def _extract_doc_text(doc_bytes: bytes) -> str:
    # Best-effort fallback for legacy .doc files when dedicated parsers are unavailable.
    candidates: List[str] = []
    for encoding in ("utf-16le", "utf-8", "latin-1"):
        try:
            decoded = doc_bytes.decode(encoding, errors="ignore")
        except Exception:
            continue
        cleaned = _normalize_extracted_text(decoded)
        if len(cleaned) >= 60:
            candidates.append(cleaned)

    ascii_chunks = re.findall(rb"[A-Za-z0-9][\x20-\x7E]{5,}", doc_bytes)
    if ascii_chunks:
        ascii_text = " ".join(chunk.decode("latin-1", errors="ignore") for chunk in ascii_chunks)
        cleaned = _normalize_extracted_text(ascii_text)
        if len(cleaned) >= 60:
            candidates.append(cleaned)

    if not candidates:
        raise ValueError("DOC unreadable or empty")
    return max(candidates, key=len)


def _extract_text_from_uploaded_document(file_bytes: bytes, kind: str) -> str:
    if kind == "pdf":
        return extract_pdf_text(file_bytes)
    if kind == "txt":
        return _extract_txt_text(file_bytes)
    if kind == "docx":
        return _extract_docx_text(file_bytes)
    if kind == "doc":
        return _extract_doc_text(file_bytes)
    raise ValueError(f"Unsupported attachment type: {kind}")


def _option_labels(options: Any) -> List[str]:
    if not isinstance(options, list):
        return []
    labels: List[str] = []
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).strip()
        else:
            label = str(opt).strip()
        if label:
            labels.append(label)
    return labels


def _node_option_labels(node: Optional[dict]) -> List[str]:
    if not isinstance(node, dict):
        return []
    return _option_labels(node.get("options", []))


def _extract_customer_name_from_text(pdf_text: str) -> Optional[str]:
    if not isinstance(pdf_text, str) or not pdf_text.strip():
        return None

    patterns = [
        r"(?:customer|client)\s*name\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,80})",
        r"(?:name of customer|customer)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,80})",
    ]
    for pattern in patterns:
        m = re.search(pattern, pdf_text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _customer_name_from_session(session: dict) -> Optional[str]:
    name = session.get("customer_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _build_pdf_context_for_export(session: dict) -> dict:
    base = session.get("product_contexts") or session.get("context") or {}
    if not isinstance(base, dict):
        base = {}
    export_ctx = dict(base)
    customer_name = _customer_name_from_session(session)
    if customer_name:
        export_ctx["__customer_name"] = customer_name
    queries = _collect_queries_for_summary(session)
    if queries:
        export_ctx["__queries"] = queries
    return export_ctx


def _has_value(entry: Optional[dict]) -> bool:
    if not isinstance(entry, dict):
        return False
    return entry.get("value") not in (None, "", [])


def _service_key_from_context(context: dict) -> Optional[str]:
    service_entry = context.get("service_select")
    if not isinstance(service_entry, dict):
        return None

    value = service_entry.get("value")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None

    val = value.strip().lower()
    if "port" in val:
        return "porting"
    if "optim" in val:
        return "optimization"
    if "audio" in val and "app" in val:
        return "audio_app"
    return None


def _entry_value_to_text(entry: Optional[dict]) -> str:
    if not isinstance(entry, dict):
        return "N/A"
    value = entry.get("value")
    if value in (None, "", []):
        return "N/A"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _extract_query_values_from_context(ctx: Any) -> List[str]:
    if not isinstance(ctx, dict):
        return []
    query_entry = ctx.get("query_input")
    if not isinstance(query_entry, dict):
        return []
    values = query_entry.get("value")
    if isinstance(values, list):
        return [str(v).strip() for v in values if str(v).strip()]
    if isinstance(values, str) and values.strip():
        return [values.strip()]
    return []


def _collect_queries_for_summary(session: dict) -> List[str]:
    queries: List[str] = []

    for q in _extract_query_values_from_context(session.get("context")):
        queries.append(q)

    product_contexts = session.get("product_contexts")
    if isinstance(product_contexts, dict):
        for product_name in session.get("product_order") or list(product_contexts.keys()):
            for q in _extract_query_values_from_context(product_contexts.get(product_name)):
                queries.append(q)

    return queries


def _build_numbered_summary(session: dict) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    lines: List[str] = []
    index_map: Dict[int, Dict[str, Any]] = {}
    number = 1

    product_contexts = session.get("product_contexts")
    if isinstance(product_contexts, dict) and product_contexts:
        product_order = session.get("product_order") or list(product_contexts.keys())
        for p_index, product_name in enumerate(product_order, start=1):
            product_ctx = product_contexts.get(product_name, {})
            if not isinstance(product_ctx, dict):
                product_ctx = {}
            start_number = number
            requirement_lines: List[str] = []
            for qid in _relevant_requirement_ids(product_ctx):
                question_text = QUESTIONS.get(qid, {}).get("text", qid)
                value_text = _entry_value_to_text(product_ctx.get(qid))
                requirement_lines.append(f"{number}. ({product_name}) {question_text} -> {value_text}")
                index_map[number] = {
                    "kind": "requirement",
                    "product": product_name,
                    "qid": qid,
                }
                number += 1
            end_number = number - 1
            if requirement_lines:
                lines.append(
                    f"Product {p_index}: {product_name} (Requirement numbers: {start_number}-{end_number})"
                )
                # Keep a blank line before numbered items so markdown renders
                # ordered lists even when numbers do not start at 1.
                lines.append("")
                lines.extend(requirement_lines)
            else:
                lines.append(f"Product {p_index}: {product_name} (No requirements captured)")
            lines.append("")
    else:
        context = session.get("context", {})
        lines.append("Product 1: default")
        for qid in _relevant_requirement_ids(context):
            question_text = QUESTIONS.get(qid, {}).get("text", qid)
            value_text = _entry_value_to_text(context.get(qid))
            lines.append(f"{number}. {question_text} -> {value_text}")
            index_map[number] = {
                "kind": "requirement",
                "product": None,
                "qid": qid,
            }
            number += 1
        lines.append("")

    queries = _collect_queries_for_summary(session)
    if queries:
        lines.append("Queries")
        for idx, query in enumerate(queries):
            lines.append(f"{number}. Query -> {query}")
            index_map[number] = {"kind": "query", "query_index": idx}
            number += 1

    summary_text = "\n".join(line for line in lines if line is not None).strip()
    if not summary_text:
        summary_text = "No requirements captured yet."
    return summary_text, index_map


def _set_query_by_index(session: dict, query_index: int, new_value: str) -> bool:
    if query_index < 0:
        return False
    remaining = query_index

    context = session.get("context")
    if isinstance(context, dict):
        query_entry = context.get("query_input")
        if isinstance(query_entry, dict) and isinstance(query_entry.get("value"), list):
            values = query_entry["value"]
            if remaining < len(values):
                values[remaining] = new_value
                return True
            remaining -= len(values)

    product_contexts = session.get("product_contexts")
    if isinstance(product_contexts, dict):
        for product_name in session.get("product_order") or list(product_contexts.keys()):
            product_ctx = product_contexts.get(product_name)
            if not isinstance(product_ctx, dict):
                continue
            query_entry = product_ctx.get("query_input")
            if not isinstance(query_entry, dict) or not isinstance(query_entry.get("value"), list):
                continue
            values = query_entry["value"]
            if remaining < len(values):
                values[remaining] = new_value
                return True
            remaining -= len(values)
    return False


def _relevant_requirement_ids(context: dict) -> List[str]:
    service_key = _service_key_from_context(context)
    excluded = SERVICE_EXCLUDED_REQUIREMENTS.get(service_key, set())
    return [qid for qid in REQUIREMENT_IDS_IN_ORDER if qid not in excluded]


def _next_unanswered_requirement_id(context: dict) -> Optional[str]:
    for qid in _relevant_requirement_ids(context):
        if not _has_value(context.get(qid)):
            return qid
    return None


def _set_active_product(session: dict, product_name: Optional[str]) -> None:
    if not product_name:
        return
    product_contexts = session.get("product_contexts")
    if not isinstance(product_contexts, dict):
        return
    if product_name not in product_contexts:
        return
    if not isinstance(product_contexts[product_name], dict):
        product_contexts[product_name] = {}
    session["active_product"] = product_name
    session["context"] = product_contexts[product_name]


def _active_product_context(session: dict) -> dict:
    product_contexts = session.get("product_contexts")
    active_product = session.get("active_product")
    if isinstance(product_contexts, dict) and active_product in product_contexts:
        ctx = product_contexts.get(active_product)
        if not isinstance(ctx, dict):
            ctx = {}
            product_contexts[active_product] = ctx
        session["context"] = ctx
        return ctx
    return session.setdefault("context", {})


def _next_missing_product_and_qid(session: dict) -> Tuple[Optional[str], Optional[str]]:
    product_contexts = session.get("product_contexts")
    if isinstance(product_contexts, dict) and product_contexts:
        product_order = session.get("product_order") or list(product_contexts.keys())
        for product_name in product_order:
            ctx = product_contexts.get(product_name, {})
            if not isinstance(ctx, dict):
                continue
            next_qid = _next_unanswered_requirement_id(ctx)
            if next_qid:
                return product_name, next_qid
        return None, None

    return None, _next_unanswered_requirement_id(session.get("context", {}))


def _requirement_prompt_prefix(session: dict, next_qid: Optional[str]) -> str:
    if not next_qid or next_qid not in ALLOWED_QUESTION_IDS:
        return ""

    product_contexts = session.get("product_contexts")
    if not isinstance(product_contexts, dict) or len(product_contexts) <= 1:
        return ""

    active_product = session.get("active_product")
    if not active_product:
        return ""

    announced_products = session.get("announced_products")
    if not isinstance(announced_products, list):
        announced_products = []

    if active_product in announced_products:
        return ""

    announced_products.append(active_product)
    session["announced_products"] = announced_products
    return f"Asking remaining requirements for {active_product}.\n"

async def handle_guest_chat(user_id: str, message: str):
    guest_session = user_sessions.get(user_id, {"messages": []})
    guest_session["messages"].append({"role": "user", "content": message})

    async def _guest_reply(text: str) -> str:
        parts = await generate_answer_async(text)
        if isinstance(parts, list) and parts:
            return parts[0]
        return parts if isinstance(parts, str) else str(parts)

    reply_text = await _guest_reply(message)
    guest_session["messages"].append({"role": "assistant", "content": reply_text})
    user_sessions[user_id] = guest_session

    return {
        "reply": reply_text,
        "guest": True,
        "in_flow": False,
        "session_id": user_id,
    }


async def init_or_get_session(req: ChatRequest, user_id: str):
    raw_session_id = req.session_id
    session_key = f"{user_id}:{raw_session_id}" if raw_session_id else user_id

    default_session = {
        "in_flow": False,
        "node_id": None,
        "context": {},
        "product_contexts": {},
        "product_order": [],
        "active_product": None,
        "announced_products": [],
        "customer_name": None,
        "pending_after_customer": None,
        "pending_attachment": None,
        "change_target": None,
        "summary_index_map": {},
        "greeting_sent": False,
        "authenticated": False,
        "session_id": raw_session_id,
        "pdf_mode": False,
        "pdf_processed": False,
    }

    session = user_sessions.get(session_key, {})
    for k, v in default_session.items():
        session.setdefault(k, v)

    user_sessions[session_key] = session
    return session, session_key, session.get("session_id")


async def get_llm_reply(text: str) -> str:
    if not text or not text.strip():
        return ""
    parts = await generate_answer_async(text)
    if isinstance(parts, list) and parts:
        return parts[0]
    return parts if isinstance(parts, str) else str(parts)


async def handle_greeting(
    message: str,
    user_id: str,
    session_key: str,
    session: dict,
    session_id: str,
):
    if session.get("greeting_sent"):
        reply_text = "How can I help you with your requirements today?"
    else:
        reply_text = GREETING_REPLY
        session["greeting_sent"] = True

    await persist_chat_pair(
            user_id, session_key, session, session_id, message, reply_text
        )

    user_sessions[session_key] = session

    return {
        "reply": reply_text,
        "in_flow": False,
        "session_id": session_id,
    }


async def handle_service_trigger(
    message: str,
    user_id: str,
    session: dict,
    session_key: str,
    session_id: str,
):
    session.update(
        {
            "in_flow": True,
            "node_id": "start",
            "context": {},
            "product_contexts": {},
            "product_order": [],
            "active_product": None,
            "announced_products": [],
            "customer_name": None,
            "pending_after_customer": None,
            "pending_attachment": None,
            "pdf_mode": True,
            "pdf_processed": False,
        }
    )

    user_sessions[session_key] = session

    start_node = conversation_flow["start"]
    reply_text = start_node["text"]
    options = _node_option_labels(start_node)

    update_session_title_if_needed(
        session=session,
        session_id=session_id,
        message=message,     # service trigger text
        in_flow=True,
    )

    await persist_chat_pair(
        user_id=user_id,
        session_key=session_key,
        session=session,
        session_id=session_id,
        user_message=message,      
        assistant_message=reply_text,
    )

    return {
        "reply": reply_text,
        "options": options,
        "in_flow": True,
        "node_id": "start",
    }




async def handle_normal_qa(
    message,
    session,
    session_key,
    user_id,
    session_id,
):
    async def _get_reply_text(text: str) -> str:
        if not text or not text.strip():
            return ""
        parts = await generate_answer_async(text)
        if isinstance(parts, list) and parts:
            return parts[0]
        return parts if isinstance(parts, str) else str(parts)

    # Persist user message immediately so UI can show it even while model is still processing.
    try:
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=message,
            assistant_message=None,
        )
    except Exception:
        logger.exception("Supabase save failed during normal chat (user immediate persist).")

    reply_text = await _get_reply_text(message)

    try:
        session_id = await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=None,
            assistant_message=reply_text,
        )
    except Exception:
        logger.exception("Supabase save failed during normal chat (assistant persist).")

    return {
        "reply": reply_text,
        "in_flow": False,
        "session_id": session_id,
    }

async def handle_pdf_upload_and_extraction(
    req: ChatRequest,
    session: dict,
    session_key: str,
    user_id: str,
    session_id: str,
    message: str,
):
    original_file_name = (req.attachment.get("name") if isinstance(req.attachment, dict) else None) or "uploaded_file"
    original_file_name = str(original_file_name).strip() or "uploaded_file"
    user_message_for_persist = (message or "").strip() or original_file_name

    attachment_kind = _infer_attachment_kind(req.attachment)
    if not attachment_kind:
        reply_text = (
            "Unsupported file type.\n\n"
            "Please upload a PDF, DOC, DOCX, or TXT file, or type 'No PDF' to continue without one."
        )
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=user_message_for_persist,
            assistant_message=reply_text,
        )
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "pdf_upload",
            "context": session["context"],
        }

    encoded_file = req.attachment.get("bytes") if isinstance(req.attachment, dict) else None
    if not isinstance(encoded_file, str) or not encoded_file.strip():
        reply_text = (
            "The uploaded file is missing content.\n\n"
            "Please upload a valid PDF, DOC, DOCX, or TXT file, or type 'No PDF' to continue without one."
        )
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=user_message_for_persist,
            assistant_message=reply_text,
        )
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "pdf_upload",
            "context": session["context"],
        }

    encoded_file = encoded_file.strip()
    if encoded_file.lower().startswith("data:") and "," in encoded_file:
        encoded_file = encoded_file.split(",", 1)[1]

    try:
        file_bytes = base64.b64decode(encoded_file, validate=True)
    except (ValueError, binascii.Error):
        reply_text = (
            "The uploaded file could not be decoded.\n\n"
            "Please upload a valid PDF, DOC, DOCX, or TXT file, or type 'No PDF' to continue without one."
        )
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=user_message_for_persist,
            assistant_message=reply_text,
        )
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "pdf_upload",
            "context": session["context"],
        }

    try:
        extracted_text = _extract_text_from_uploaded_document(file_bytes, attachment_kind)
    except (PDFParseError, ValueError) as exc:
        logger.warning(
            "Invalid upload for session_id=%s file=%s kind=%s: %s",
            session_id,
            original_file_name,
            attachment_kind,
            exc,
        )
        reply_text = (
            "The uploaded file is not readable.\n\n"
            "Please upload a readable PDF, DOC, DOCX, or TXT document containing your requirements, "
            "or type 'No PDF' to continue without one."
        )
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=user_message_for_persist,
            assistant_message=reply_text,
        )
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "pdf_upload",
            "context": session["context"],
        }

    file_url = upload_file_to_supabase(file_bytes, original_file_name)
    user_attachment = {"type": attachment_kind, "name": original_file_name, "url": file_url}
    await persist_chat_pair(
        user_id=user_id,
        session_key=session_key,
        session=session,
        session_id=session_id,
        user_message=user_message_for_persist,
        assistant_message=None,
        user_attachment=user_attachment,
    )
    user_message_for_persist = None

    extracted_customer_name = _extract_customer_name_from_text(extracted_text)
    if extracted_customer_name:
        session["customer_name"] = extracted_customer_name
        logger.info("[PDF-MERGE] Customer name extracted: %s", extracted_customer_name)

    extracted = await extract_answers_from_pdf(
        pdf_text=extracted_text,
        questions=QUESTIONS,
    )

    # -----------------------------------
    # FILTER TO ALLOWED QUESTION IDS ONLY
    # -----------------------------------

    cleaned = {}

    for product, answers in extracted.items():
        if not isinstance(answers, dict):
            continue

        for qid, obj in answers.items():
            if qid not in ALLOWED_QUESTION_IDS:
                continue

            cleaned.setdefault(product, {})[qid] = obj

    for product, answers in cleaned.items():
        for qid, obj in answers.items():
            if not isinstance(obj, dict):
                continue
            if obj.get("value") in (None, "", []):
                continue

            logger.info(
                "[PDF-MERGE] Extracted product=%s qid=%s value=%r confidence=%.3f evidence=%s",
                product,
                qid,
                obj.get("value"),
                float(obj.get("confidence", 0.0)),
                obj.get("evidence", "")[:200],
            )


    # -----------------------------------
    # MERGE EXTRACTION INTO PRODUCT CONTEXTS
    # -----------------------------------

    if not cleaned:
        cleaned = {"default": {}}

    product_contexts: Dict[str, Dict[str, Any]] = {}

    for product, answers in cleaned.items():
        product_contexts[product] = {}
        for qid in REQUIREMENT_IDS_IN_ORDER:
            obj = answers.get(qid)
            if not isinstance(obj, dict):
                continue
            if obj.get("value") in (None, "", []):
                continue

            product_contexts[product][qid] = {
                "value": obj["value"],
                "confidence": float(obj.get("confidence", 0.0)),
                "source": "pdf",
                "product": product,
                "evidence": obj.get("evidence"),
            }
            logger.info(
                "[PDF-MERGE] Accepted product=%s qid=%s value=%r confidence=%.3f",
                product,
                qid,
                obj.get("value"),
                float(obj.get("confidence", 0.0)),
            )

    session["product_contexts"] = product_contexts
    session["product_order"] = list(product_contexts.keys())
    session["announced_products"] = []

    if session["product_order"]:
        _set_active_product(session, session["product_order"][0])
    else:
        session["active_product"] = None
        session["context"] = {}

    active_product, next_node = _next_missing_product_and_qid(session)
    if active_product:
        _set_active_product(session, active_product)

    session["pdf_processed"] = True
    session["pdf_mode"] = False
    user_sessions[session_key] = session

    # -------- PARTIAL PDF (always generated) --------
    pdf_context = _build_pdf_context_for_export(session)
    partial_pdf = generate_final_requirements_pdf(
        context=pdf_context,
        questions=QUESTIONS,
    )

    partial_url = upload_pdf_to_supabase(
        partial_pdf,
        f"REQ_PARTIAL_{session_id}.pdf",
    )

    attachment = {
        "type": "pdf",
        "name": "Extracted Requirements.pdf",
        "url": partial_url,
    }

    # -------- CASE 1: ALL FOUND --------
    if not next_node:
        if not _customer_name_from_session(session):
            session["node_id"] = "__customer_name_input"
            session["in_flow"] = True
            session["pending_after_customer"] = "__complete_pdf_extraction"
            session["pending_attachment"] = attachment
            user_sessions[session_key] = session

            ask_name_reply = "Please enter the customer name before finalizing the requirements."
            await persist_chat_pair(
                user_id, session_key, session, session_id, user_message_for_persist, ask_name_reply
            )
            return {
                "reply": ask_name_reply,
                "in_flow": True,
                "node_id": "__customer_name_input",
                "context": session["context"],
            }

        reply = (
            "✅ Requirements Received – Complete\n\n"
            "Your requirements have been collected successfully.\n\n"
            "All necessary details have been provided. Our team will proceed with further analysis."
        )

        await persist_chat_pair(
            user_id, session_key, session, session_id, user_message_for_persist, reply, attachment
        )

        return {
            "reply": reply,
            "attachment": attachment,
            "in_flow": False,
            "context": session["context"],
        }

    if not next_node or next_node not in conversation_flow:
        # All missing fields are already filled by extraction
        # Flow is NOT finished — move to final_step

        session["node_id"] = "final_step"
        session["in_flow"] = True
        user_sessions[session_key] = session

        final_node = conversation_flow["final_step"]
        reply_text = final_node["text"]

        # Save PDF + transition message
        await persist_chat_pair(
            user_id,
            session_key,
            session,
            session_id,
            user_message_for_persist,
            reply_text,
            attachment=attachment,
        )

        return {
            "reply": reply_text,
            "attachment": attachment,
            "options": _node_option_labels(final_node),
            "in_flow": True,
            "node_id": "final_step",
            "context": session["context"],
        }

    session["node_id"] = next_node
    user_sessions[session_key] = session

    status_reply = (
        "📌 Requirements Received – Additional Details Needed\n\n"
        "Thank you for submitting your requirements.\n\n"
        "We have successfully collected and reviewed the provided information. "
        "However, certain details appear to be incomplete or require further clarification "
        "in order to proceed accurately.\n\n"
        "Kindly review the extracted requirements below and provide the remaining "
        "information by answering the questions listed underneath. Your responses will "
        "help us ensure a precise analysis and prevent any potential delays in implementation.\n\n"
        "If you need any clarification regarding the questions, please feel free to reach out. "
        "We are happy to assist."
    )

    await persist_chat_pair(
        user_id,
        session_key,
        session,
        session_id,
        user_message_for_persist,
        status_reply,
        attachment=attachment,
    )

    question_node = conversation_flow[next_node]
    question_text = f"{_requirement_prompt_prefix(session, next_node)}{question_node.get('text', '')}"
    options = question_node.get("options", [])

    # Persist the first follow-up question so Supabase chat history
    # includes what the UI asks immediately after the status message.
    if question_text.strip():
        await persist_chat_pair(
            user_id=user_id,
            session_key=session_key,
            session=session,
            session_id=session_id,
            user_message=None,
            assistant_message=question_text,
        )

    # RETURN TWO DISTINCT MESSAGES
    return {
        "reply": status_reply,
        "attachment": attachment,
        "followup": {
            "question": question_text,
            "options": _option_labels(options),
        },
        "in_flow": True,
        "node_id": next_node,
        "context": session["context"],
    }


def _next_requirement_or_final(session: dict) -> str:
    next_product, next_id = _next_missing_product_and_qid(session)
    if next_product:
        _set_active_product(session, next_product)
    if next_id and next_id in conversation_flow:
        return next_id
    return "final_step"


async def handle_flow_engine(
    message: str,
    session: dict,
    session_key: str,
    user_id: str,
    session_id: str,
):
    current_id = session["node_id"]
    node = conversation_flow.get(current_id)

    if not session.get("in_flow"):
        reply = await get_llm_reply(message)
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply)
        return {"reply": reply, "in_flow": False}

    if current_id == "__customer_name_input":
        entered_name = message.strip()
        if not entered_name:
            reply_text = "Customer name cannot be empty. Please enter the customer name."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "in_flow": True,
                "node_id": "__customer_name_input",
            }

        session["customer_name"] = entered_name
        pending_after_customer = session.get("pending_after_customer")
        pending_attachment = session.get("pending_attachment")
        session["pending_after_customer"] = None
        session["pending_attachment"] = None

        if pending_after_customer == "__complete_pdf_extraction":
            session["in_flow"] = False
            session["node_id"] = None
            user_sessions[session_key] = session

            reply_text = (
                "Your requirements have been collected successfully.\n\n"
                "Please review them below."
            )
            await persist_chat_pair(
                user_id,
                session_key,
                session,
                session_id,
                message,
                reply_text,
                pending_attachment,
            )
            return {
                "reply": reply_text,
                "attachment": pending_attachment,
                "in_flow": False,
                "context": session["context"],
            }

        if pending_after_customer == "submit_response":
            pdf_context = _build_pdf_context_for_export(session)
            pdf_bytes = generate_final_requirements_pdf(
                context=pdf_context,
                questions=QUESTIONS,
            )
            pdf_url = upload_pdf_to_supabase(pdf_bytes, f"REQ_{session_id}.pdf")
            attachment = {
                "type": "pdf",
                "name": "Requirements.pdf",
                "url": pdf_url,
            }
            reply_text = "Thank you! Your requirements have been submitted."

            await persist_chat_pair(
                user_id, session_key, session, session_id, message, reply_text, attachment
            )

            session["in_flow"] = False
            session["node_id"] = None
            user_sessions[session_key] = session
            return {
                "reply": reply_text,
                "attachment": attachment,
                "in_flow": False,
            }

        next_id = pending_after_customer if pending_after_customer in conversation_flow else "final_step"
        session["node_id"] = next_id
        session["in_flow"] = True
        user_sessions[session_key] = session

        next_node = conversation_flow.get(next_id, {})
        reply_text = next_node.get("text", "Customer name saved.")
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": _node_option_labels(next_node),
            "in_flow": True,
            "node_id": next_id,
            "context": session["context"],
        }

    # mistake_prompt
    if current_id == "mistake_prompt":
        ans = message.strip().lower()

        if ans in ("yes", "y"):
            last_q = session.get("last_question_node")
            if not last_q or last_q not in conversation_flow:
                session["in_flow"] = False
                session["node_id"] = None
                user_sessions[session_key] = session
                reply_text = "Unable to continue the flow from the previous step."
                await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
                return {"reply": reply_text, "in_flow": False}

            session["node_id"] = last_q
            user_sessions[session_key] = session

            q_node = conversation_flow[last_q]
            reply_text = q_node["text"]

            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "options": _node_option_labels(q_node),
                "in_flow": True,
                "node_id": last_q,
                "context": session["context"],
            }

        if ans in ("no", "n"):
            session["in_flow"] = False
            session["node_id"] = None
            user_sessions[session_key] = session

            reply_text = "No problem, ending the requirements flow."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {"reply": reply_text, "in_flow": False}

        reply_text = "Please type Yes or No."
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": ["Yes", "No"],
            "in_flow": True,
            "node_id": "mistake_prompt",
        }

    if current_id == "make_changes":
        raw = message.strip()
        if not raw.isdigit():
            reply_text = "Please enter a valid requirement number from the summary."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {"reply": reply_text, "in_flow": True, "node_id": "make_changes"}

        selected_number = int(raw)
        summary_index_map = session.get("summary_index_map") or {}
        target = summary_index_map.get(selected_number)
        if not target:
            reply_text = "That number is not in the summary. Please enter a valid number."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {"reply": reply_text, "in_flow": True, "node_id": "make_changes"}

        session["change_target"] = target
        session["node_id"] = "await_new_answer"
        session["in_flow"] = True
        user_sessions[session_key] = session

        reply_text = conversation_flow["await_new_answer"]["text"]
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "await_new_answer",
            "context": session["context"],
        }

    if current_id == "await_new_answer":
        new_value = message.strip()
        if not new_value:
            reply_text = "New value cannot be empty. Please enter a valid value."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {"reply": reply_text, "in_flow": True, "node_id": "await_new_answer"}

        target = session.get("change_target")
        if not isinstance(target, dict):
            reply_text = "Unable to identify the selected field. Please select a number again."
            session["node_id"] = "make_changes"
            user_sessions[session_key] = session
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {"reply": reply_text, "in_flow": True, "node_id": "make_changes"}

        if target.get("kind") == "requirement":
            qid = target.get("qid")
            product = target.get("product")
            if product and isinstance(session.get("product_contexts"), dict):
                product_ctx = session["product_contexts"].setdefault(product, {})
                product_ctx[qid] = {"value": new_value, "confidence": 1.0, "source": "user"}
                if session.get("active_product") == product:
                    session["context"] = product_ctx
            else:
                ctx = session.setdefault("context", {})
                ctx[qid] = {"value": new_value, "confidence": 1.0, "source": "user"}
        elif target.get("kind") == "query":
            if not _set_query_by_index(session, int(target.get("query_index", -1)), new_value):
                ctx = session.setdefault("context", {})
                q_entry = ctx.setdefault(
                    "query_input",
                    {"value": [], "confidence": 1.0, "source": "user"},
                )
                if not isinstance(q_entry.get("value"), list):
                    q_entry["value"] = []
                q_entry["value"].append(new_value)

        summary_text, index_map = _build_numbered_summary(session)
        session["summary_index_map"] = index_map
        session["change_target"] = None
        session["node_id"] = "show_updated_summary"
        session["in_flow"] = True
        user_sessions[session_key] = session

        updated_node = conversation_flow["show_updated_summary"]
        reply_text = updated_node["text"].replace("{{updated_summary}}", summary_text)
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": _node_option_labels(updated_node),
            "in_flow": True,
            "node_id": "show_updated_summary",
            "context": session["context"],
        }

    # missing node safety
    if not node:
        session["in_flow"] = False
        user_sessions[session_key] = session

        safe_reply = await get_llm_reply(message)
        await persist_chat_pair(user_id, session_key, session, session_id, message, safe_reply)
        return {"reply": safe_reply, "in_flow": False}

    # start node: explicit routing, no next-pointer dependency
    if current_id == "start":
        selected = flow_manager.find_best_option(node.get("options", []), message)
        if not selected:
            session["node_id"] = "mistake_prompt"
            session["last_question_node"] = current_id
            user_sessions[session_key] = session

            reply_text = "That did not match any available option."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "options": ["Yes", "No"],
                "in_flow": True,
                "node_id": "mistake_prompt",
            }

        session["context"]["start"] = {
            "value": selected["label"],
            "confidence": 1.0,
            "source": "user",
        }

        selected_label = selected["label"].strip().lower()
        if selected_label.startswith("yes"):
            next_id = "pdf_upload"
            session["product_contexts"] = {}
            session["product_order"] = []
            session["active_product"] = None
            session["announced_products"] = []
            session["pdf_mode"] = True
            session["pdf_processed"] = False
        else:
            next_id = "service_select"
            session["product_contexts"] = {}
            session["product_order"] = []
            session["active_product"] = None
            session["announced_products"] = []
            session["pdf_mode"] = False
            session["pdf_processed"] = True

        session["node_id"] = next_id
        session["in_flow"] = True
        user_sessions[session_key] = session

        next_node = conversation_flow.get(next_id, {})
        reply_text = next_node.get("text", "")
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)

        return {
            "reply": reply_text,
            "options": _node_option_labels(next_node),
            "in_flow": True,
            "node_id": next_id,
            "context": session["context"],
        }

    # pdf_upload node: allow continue without PDF
    if current_id == "pdf_upload":
        user_text = message.strip().lower()
        if user_text in PDF_SKIP_INPUTS:
            next_id = "service_select"
            session["product_contexts"] = {}
            session["product_order"] = []
            session["active_product"] = None
            session["announced_products"] = []
            session["pdf_mode"] = False
            session["pdf_processed"] = True
            session["node_id"] = next_id
            user_sessions[session_key] = session

            next_node = conversation_flow[next_id]
            reply_text = next_node.get("text", "")
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "options": _node_option_labels(next_node),
                "in_flow": True,
                "node_id": next_id,
                "context": session["context"],
            }

        reply_text = (
            "Please upload a PDF, DOC, DOCX, or TXT document containing your requirements.\n\n"
            "If you want to continue without a file, type 'No PDF'."
        )
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "in_flow": True,
            "node_id": "pdf_upload",
            "context": session["context"],
        }

    # requirement nodes: ask by remaining IDs in order (service-aware)
    if current_id in ALLOWED_QUESTION_IDS:
        current_context = _active_product_context(session)

        if node.get("options"):
            selected = flow_manager.find_best_option(node["options"], message)
            if not selected:
                session["node_id"] = "mistake_prompt"
                session["last_question_node"] = current_id
                user_sessions[session_key] = session

                reply_text = "That did not match any available option."
                await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
                return {
                    "reply": reply_text,
                    "options": ["Yes", "No"],
                    "in_flow": True,
                    "node_id": "mistake_prompt",
                }
            answer_value = selected["label"]
        else:
            answer_value = message.strip()

        current_context[current_id] = {
            "value": answer_value,
            "confidence": 1.0,
            "source": "user",
        }
        session["context"] = current_context
        if isinstance(session.get("product_contexts"), dict) and session.get("active_product") in session["product_contexts"]:
            session["product_contexts"][session["active_product"]] = current_context

        next_id = _next_requirement_or_final(session)
        session["node_id"] = next_id
        session["in_flow"] = True
        user_sessions[session_key] = session

        next_node = conversation_flow[next_id]
        reply_text = f"{_requirement_prompt_prefix(session, next_id)}{next_node['text']}"

        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": _node_option_labels(next_node),
            "in_flow": True,
            "node_id": next_id,
            "context": session["context"],
        }

    # non-requirement input nodes (query flow etc.)
    if node.get("expect_user_input") or node.get("type") == "input":
        node_id = node["id"]

        if node_id == "query_input":
            existing = session["context"].get(node_id)
            if not existing:
                session["context"][node_id] = {
                    "value": [message.strip()],
                    "confidence": 1.0,
                    "source": "user",
                }
            else:
                existing["value"].append(message.strip())
        else:
            session["context"][node_id] = {
                "value": message.strip(),
                "confidence": 1.0,
                "source": "user",
            }

        next_id = node.get("next") or "final_step"

        session["node_id"] = next_id
        session["in_flow"] = True
        user_sessions[session_key] = session

        if next_id not in conversation_flow:
            session["in_flow"] = False
            session["node_id"] = None
            user_sessions[session_key] = session

            reply = await get_llm_reply(message)
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply)
            return {"reply": reply, "in_flow": False}

        next_node = conversation_flow[next_id]
        reply_text = next_node["text"]

        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": _node_option_labels(next_node),
            "in_flow": True,
            "node_id": next_id,
            "context": session["context"],
        }

    # non-requirement option nodes (summary/query step flow)
    if node.get("options"):
        selected = flow_manager.find_best_option(node["options"], message)
        if not selected:
            session["node_id"] = "mistake_prompt"
            session["last_question_node"] = current_id
            user_sessions[session_key] = session

            reply_text = "That did not match any available option."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "options": ["Yes", "No"],
                "in_flow": True,
                "node_id": "mistake_prompt",
            }

        session["context"][node["id"]] = {
            "value": selected["label"],
            "confidence": 1.0,
            "source": "user",
        }
        next_id = selected.get("next")
    else:
        next_id = node.get("next")

    session["node_id"] = next_id
    session["in_flow"] = True
    user_sessions[session_key] = session

    if next_id not in conversation_flow:
        session["in_flow"] = False
        session["node_id"] = None
        user_sessions[session_key] = session

        reply = await get_llm_reply(message)
        await persist_chat_pair(user_id, session_key, session, session_id, message, reply)
        return {"reply": reply, "in_flow": False}

    next_node = conversation_flow[next_id]

    if next_id in ("show_summary", "show_updated_summary"):
        summary_text, index_map = _build_numbered_summary(session)
        session["summary_index_map"] = index_map
        placeholder = "{{summary}}" if next_id == "show_summary" else "{{updated_summary}}"
        reply_text = next_node["text"].replace(placeholder, summary_text)

        await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
        return {
            "reply": reply_text,
            "options": _node_option_labels(next_node),
            "in_flow": True,
            "node_id": next_id,
        }

    if next_id == "submit_response":
        if not _customer_name_from_session(session):
            session["node_id"] = "__customer_name_input"
            session["in_flow"] = True
            session["pending_after_customer"] = "submit_response"
            user_sessions[session_key] = session

            reply_text = "Please enter the customer name before submission."
            await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)
            return {
                "reply": reply_text,
                "in_flow": True,
                "node_id": "__customer_name_input",
                "context": session["context"],
            }

        pdf_context = _build_pdf_context_for_export(session)

        pdf_bytes = generate_final_requirements_pdf(
            context=pdf_context,
            questions=QUESTIONS,
        )
        pdf_url = upload_pdf_to_supabase(pdf_bytes, f"REQ_{session_id}.pdf")

        attachment = {
            "type": "pdf",
            "name": "Requirements.pdf",
            "url": pdf_url,
        }

        reply_text = "Thank you! Your requirements have been submitted."

        await persist_chat_pair(
            user_id, session_key, session, session_id, message, reply_text, attachment
        )

        session["in_flow"] = False
        session["node_id"] = None
        user_sessions[session_key] = session

        return {
            "reply": reply_text,
            "attachment": attachment,
            "in_flow": False,
        }

    reply_text = next_node["text"]
    await persist_chat_pair(user_id, session_key, session, session_id, message, reply_text)

    for k, v in session["context"].items():
        assert isinstance(v, dict), f"Context[{k}] is not dict: {type(v)}"
        assert "value" in v, f"Context[{k}] missing 'value'"
        assert "confidence" in v, f"Context[{k}] missing 'confidence'"
        assert "source" in v, f"Context[{k}] missing 'source'"

    return {
        "reply": reply_text,
        "options": _node_option_labels(next_node),
        "in_flow": True,
        "node_id": next_id,
        "context": session["context"],
    }









