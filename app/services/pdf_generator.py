import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services.pdf_question_extractor import ALLOWED_QUESTION_IDS

PDF_LABELS = {
    "service_select": "Service Type",
    "Optimization_type": "Type of Optimization",
    "App_type": "App Type",
    "Porting_question_1": "Current Target Platform",
    "Porting_question_2": "Cross Compliance Requirement",
    "DSP_Processor": "Type of processor",
    "Application": "Application Type",
    "Audio_Interface": "Type of audio interface",
    "Audio_Params_1": "PCM Sample Size",
    "Audio_Params_2": "Sampling Frequency",
    "Audio_Params_3": "Audio Format",
    "Audio_Tech_1": "Audio processing modules",
    "Audio_Tech_2": "Current platforms supported",
    "CodeBase_1": "Languages Used",
    "CodeBase_2": "Type of Sample App",
    "CodeBase_3": "Code Size",
    "CodeBase_4": "Memory Requirement",
    "CodeBase_5": "Is the Source Code Available?",
    "CodeBase_6": "Code Implementation Type",
    "TargetPlatform_1": "Target Platform to Port",
    "TargetPlatform_2": "Chipset Details",
}

SERVICE_EXCLUDED_REQUIREMENTS = {
    "porting": {"Optimization_type", "App_type"},
    "optimization": {"Porting_type", "Porting_question_1", "Porting_question_2", "App_type", "TargetPlatform_1", "TargetPlatform_2"},
    "audio_app": {"Optimization_type", "Porting_type", "Porting_question_1", "Porting_question_2"},
}

SUMMARY_FIELDS: List[Tuple[str, str]] = [
    ("Service", "service_select"),
    ("Optimization Type", "Optimization_type"),
    ("Application Type", "App_type"),
    ("Current Platform", "Porting_question_1"),
    ("Cross Compliance", "Porting_question_2"),
    ("Processor", "DSP_Processor"),
    ("Audio Interface", "Audio_Interface"),
    ("PCM Sample Size", "Audio_Params_1"),
    ("Sampling Frequency", "Audio_Params_2"),
    ("Audio Format", "Audio_Params_3"),
    ("Audio Processing Modules", "Audio_Tech_1"),
    ("Current Supported Platforms", "Audio_Tech_2"),
    ("Languages", "CodeBase_1"),
    ("Sample App Type", "CodeBase_2"),
    ("Code Size", "CodeBase_3"),
    ("Memory Requirement", "CodeBase_4"),
    ("Source Code Availability", "CodeBase_5"),
    ("Implementation Type", "CodeBase_6"),
    ("Target Platform", "TargetPlatform_1"),
    ("Chipset Details", "TargetPlatform_2"),
]

FREQUENCY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(k?hz)\b", re.IGNORECASE)
MEMORY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb)\b", re.IGNORECASE)
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"


def _has_value(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("value") not in (None, "", [])


def _format_value(entry: Any) -> str:
    if not _has_value(entry):
        return "N/A"

    value = entry["value"]
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _read_text_value(product_context: Dict[str, Any], qid: str) -> str:
    return _format_value(product_context.get(qid))


def _first_non_na(*values: str) -> str:
    for value in values:
        if isinstance(value, str) and value.strip() and value != "N/A":
            return value
    return "N/A"


def _read_queries(context: Dict[str, Any]) -> List[str]:
    if not isinstance(context, dict):
        return []

    raw_queries = context.get("__queries")
    if isinstance(raw_queries, list):
        return [str(q).strip() for q in raw_queries if str(q).strip()]

    query_entry = context.get("query_input")
    if isinstance(query_entry, dict):
        value = query_entry.get("value")
        if isinstance(value, list):
            return [str(q).strip() for q in value if str(q).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]

    return []


def _build_project_requirement_analysis_lines(
    product_items: List[Tuple[str, Dict[str, Any]]],
    customer_name: str,
) -> List[str]:
    lines: List[str] = [
        "Project Requirement Analysis",
        "Customer",
        customer_name,
        "",
    ]

    shortlisted_chipsets: List[str] = []
    memory_requirements: List[str] = []
    migration_reasons: List[str] = []

    for idx, (product_name, product_context) in enumerate(product_items, start=1):
        platform = _first_non_na(
            _read_text_value(product_context, "Porting_question_1"),
            _read_text_value(product_context, "Audio_Tech_2"),
        )
        processes = _read_text_value(product_context, "Audio_Tech_1")
        output = _read_text_value(product_context, "Audio_Params_3")
        migration_target = _first_non_na(
            _read_text_value(product_context, "TargetPlatform_2"),
            _read_text_value(product_context, "TargetPlatform_1"),
            _read_text_value(product_context, "DSP_Processor"),
        )

        lines.append(f"Product {idx}: {product_name}")
        lines.append("Current Status")
        if platform != "N/A":
            lines.append(f"- Platform: {platform}")
        if processes != "N/A":
            lines.append(f"- Processes: {processes}")
        if output != "N/A":
            lines.append(f"- Output: {output}")
        if platform == "N/A" and processes == "N/A" and output == "N/A":
            lines.append("- Status details: N/A")
        lines.append("")

        if migration_target != "N/A":
            lines.append("Migration Requirement")
            if platform != "N/A":
                lines.append(f"- Move from {platform} platform -> {migration_target}")
            else:
                lines.append(f"- Target: {migration_target}")
            lines.append("")

        lines.append("Technical Requirements")
        tech_rows = [
            ("Chipset", _read_text_value(product_context, "TargetPlatform_2")),
            ("OS", _read_text_value(product_context, "Application")),
            ("Audio Output", _read_text_value(product_context, "Audio_Params_3")),
            ("Bluetooth Support", _read_text_value(product_context, "Porting_question_2")),
            ("Must support", _read_text_value(product_context, "Audio_Interface")),
            ("Processing", _read_text_value(product_context, "Audio_Tech_1")),
        ]
        tech_written = False
        for label, value in tech_rows:
            if value == "N/A":
                continue
            lines.append(f"- {label}: {value}")
            tech_written = True
        if not tech_written:
            lines.append("- N/A")
        lines.append("")

        chipset_value = _read_text_value(product_context, "TargetPlatform_2")
        if chipset_value != "N/A" and chipset_value not in shortlisted_chipsets:
            shortlisted_chipsets.append(chipset_value)

        memory_value = _read_text_value(product_context, "CodeBase_4")
        if memory_value != "N/A" and memory_value not in memory_requirements:
            memory_requirements.append(memory_value)

        optimization_reason = _read_text_value(product_context, "Optimization_type")
        if optimization_reason != "N/A" and optimization_reason not in migration_reasons:
            migration_reasons.append(optimization_reason)

        service_value = _read_text_value(product_context, "service_select").lower()
        if "optim" in service_value and "Cost optimization" not in migration_reasons:
            migration_reasons.append("Cost optimization")

    if migration_reasons:
        lines.append("Migration Reason")
        for reason in migration_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    if shortlisted_chipsets:
        lines.append("Shortlisted Chipsets")
        for chip in shortlisted_chipsets:
            lines.append(f"- {chip}")
        lines.append("")

    if memory_requirements:
        lines.append("Memory Requirement")
        for mem in memory_requirements:
            lines.append(f"- {mem}")
        lines.append("")

    lines.append("Technical Observations & Considerations")
    for idx, (product_name, product_context) in enumerate(product_items, start=1):
        summary = build_summary_sections(
            product_name=product_name,
            product_context=product_context,
            customer_name=customer_name,
        )
        target = _first_non_na(
            _read_text_value(product_context, "TargetPlatform_2"),
            _read_text_value(product_context, "TargetPlatform_1"),
            _read_text_value(product_context, "DSP_Processor"),
        )
        if target != "N/A":
            lines.append(f"{idx}. {product_name} migration to {target}")
        else:
            lines.append(f"{idx}. {product_name}")
        lines.append("Need to check:")
        for observation in summary["observations"]:
            lines.append(f"- {observation}")
        lines.append("")

    return lines


def _extract_max_frequency_khz(text: str) -> Optional[float]:
    if not isinstance(text, str) or not text.strip():
        return None

    values: List[float] = []
    for match in FREQUENCY_PATTERN.finditer(text):
        amount = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "hz":
            amount /= 1000.0
        values.append(amount)

    return max(values) if values else None


def _extract_memory_mb(text: str) -> Optional[float]:
    if not isinstance(text, str) or not text.strip():
        return None

    match = MEMORY_PATTERN.search(text)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "gb":
        return amount * 1024.0
    if unit == "kb":
        return amount / 1024.0
    return amount


def build_summary_sections(
    product_name: str,
    product_context: Dict[str, Any],
    customer_name: str,
) -> Dict[str, Any]:
    key_requirements = []
    for label, qid in SUMMARY_FIELDS:
        value = _read_text_value(product_context, qid)
        if value == "N/A":
            continue
        key_requirements.append(f"{label}: {value}")

    observations = []
    service = _read_text_value(product_context, "service_select").lower()
    from_platform = _read_text_value(product_context, "Porting_question_1")
    target_platform = _read_text_value(product_context, "TargetPlatform_1")
    chipset = _read_text_value(product_context, "TargetPlatform_2")
    processor = _read_text_value(product_context, "DSP_Processor")
    sampling_frequency = _read_text_value(product_context, "Audio_Params_2")
    audio_format = _read_text_value(product_context, "Audio_Params_3")
    memory_requirement = _read_text_value(product_context, "CodeBase_4")
    source_availability = _read_text_value(product_context, "CodeBase_5")
    cross_compliance = _read_text_value(product_context, "Porting_question_2")

    migration_target = next(
        (item for item in (chipset, target_platform, processor) if item != "N/A"),
        "target chipset/platform",
    )
    if "port" in service and from_platform != "N/A":
        observations.append(
            f"Migration scope is from {from_platform} to {migration_target}. Validate toolchain, audio drivers, and codec compatibility early."
        )

    max_freq_khz = _extract_max_frequency_khz(sampling_frequency)
    if max_freq_khz is not None and max_freq_khz >= 96:
        observations.append(
            f"High-rate audio requirement detected ({sampling_frequency}). Plan low-latency buffering and DSP headroom checks."
        )

    if audio_format != "N/A":
        format_lower = audio_format.lower()
        if "stereo" in format_lower or "multi" in format_lower or "channel" in format_lower:
            observations.append(
                f"Output format is {audio_format}. Confirm channel mapping, routing, and endpoint capability."
            )

    if cross_compliance != "N/A" and any(token in cross_compliance.lower() for token in ("yes", "required", "need")):
        observations.append("Cross-platform compliance is requested. Define a portability validation matrix across target environments.")

    memory_mb = _extract_memory_mb(memory_requirement)
    if memory_mb is not None and memory_mb <= 256:
        observations.append(
            f"Memory budget is about {memory_requirement}. Verify runtime allocation against OS and DSP constraints."
        )

    if source_availability != "N/A" and any(token in source_availability.lower() for token in ("no", "library", "binary")):
        observations.append(
            f"Source availability is '{source_availability}'. Confirm integration boundaries and required SDK/tool access."
        )

    if not observations:
        observations.append("Validate platform compatibility, DSP resource usage, and end-to-end audio latency with representative workloads.")

    return {
        "intro_title": "I have reviewed your uploaded requirement document:",
        "intro_label": "Requirement",
        "intro_text": (
            "Below is a clear, structured analysis and breakdown of your requirement "
            "for better understanding and further technical planning."
        ),
        "analysis_title": "Project Requirement Analysis",
        "customer": customer_name,
        "product": product_name,
        "key_requirements": key_requirements,
        "observations": observations,
    }


# def _format_confidence(entry: Any) -> str:
#     if not isinstance(entry, dict):
#         return "0.000"

#     try:
#         return f"{float(entry.get('confidence', 0.0)):.3f}"
#     except (TypeError, ValueError):
#         return "0.000"


def _read_customer_name(context: Dict[str, Any]) -> str:
    for key in ("customer_name", "customer", "name", "__customer_name"):
        value = context.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "N/A"


def _normalize_product_contexts(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(context, dict) or not context:
        return {"default": {}}

    keys = list(context.keys())
    if any(k in ALLOWED_QUESTION_IDS for k in keys):
        return {"default": context}

    product_contexts: Dict[str, Dict[str, Any]] = {}
    for product_name, product_ctx in context.items():
        if not isinstance(product_ctx, dict):
            continue
        filtered = {qid: entry for qid, entry in product_ctx.items() if qid in ALLOWED_QUESTION_IDS}
        product_contexts[str(product_name)] = filtered

    return product_contexts or {"default": {}}


def _service_key_from_product_context(product_context: Dict[str, Any]) -> str:
    entry = product_context.get("service_select")
    if not isinstance(entry, dict):
        return ""

    value = entry.get("value")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return ""

    text = value.strip().lower()
    if "port" in text:
        return "porting"
    if "optim" in text:
        return "optimization"
    if "audio" in text and "app" in text:
        return "audio_app"
    return ""


def _visible_requirement_ids(product_context: Dict[str, Any]) -> list[str]:
    service_key = _service_key_from_product_context(product_context)
    excluded = SERVICE_EXCLUDED_REQUIREMENTS.get(service_key, set())
    return [qid for qid in ALLOWED_QUESTION_IDS if qid not in excluded]


def build_pdf_rows(
    product_name: str,
    product_context: Dict[str, Any],
    questions: Dict[str, Any],
    customer_name: str,
) -> list[list[str]]:
    rows = [["RQ ID", "Requirement", "Value"]]
    rows.append(["-", "Customer Name", customer_name])
    rows.append(["-", "Product Name", product_name])

    for idx, qid in enumerate(_visible_requirement_ids(product_context), start=1):
        label = PDF_LABELS.get(qid) or questions.get(qid, {}).get("text", qid)
        entry = product_context.get(qid)
        rows.append([str(idx), label, _format_value(entry)])

    return rows


def generate_final_requirements_pdf(context: dict, questions: dict) -> bytes:
    product_contexts = _normalize_product_contexts(context or {})
    customer_name = _read_customer_name(context if isinstance(context, dict) else {})
    queries = _read_queries(context if isinstance(context, dict) else {})

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )
    styles = getSampleStyleSheet()
    report_title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=HEADER_FONT,
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F2A43"),
        spaceAfter=4,
    )
    product_style = ParagraphStyle(
        "ProductHeading",
        parent=styles["Normal"],
        fontName=HEADER_FONT,
        fontSize=12.5,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1F3B57"),
        spaceAfter=2,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=HEADER_FONT,
        fontSize=10.5,
        leading=13,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.white,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=BODY_FONT,
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#1A1A1A"),
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName=HEADER_FONT,
        fontSize=15,
        leading=19,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F2A43"),
        spaceAfter=4,
    )
    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName=HEADER_FONT,
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1F3B57"),
        spaceAfter=2,
    )
    section_body_style = ParagraphStyle(
        "SectionBody",
        parent=styles["Normal"],
        fontName=BODY_FONT,
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1A1A1A"),
    )
    elements = []

    product_items = list(product_contexts.items())

    for i, (product_name, product_context) in enumerate(product_items):
        if i > 0:
            elements.append(PageBreak())

        elements.append(Paragraph("Requirements Specification", report_title_style))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<b>Product:</b> {product_name}", product_style))
        elements.append(Spacer(1, 6))

        rows = build_pdf_rows(product_name, product_context, questions, customer_name)
        wrapped_rows = []
        for row_idx, row in enumerate(rows):
            wrapped_row = []
            for col_idx, cell in enumerate(row):
                text = "" if cell is None else str(cell)
                style = table_header_style if row_idx == 0 else table_cell_style
                if row_idx == 0:
                    text = f"<b>{text}</b>"
                if col_idx == 0:
                    style = ParagraphStyle(
                        f"{style.name}_{row_idx}_{col_idx}",
                        parent=style,
                        alignment=1,
                    )
                wrapped_row.append(Paragraph(text, style))
            wrapped_rows.append(wrapped_row)

        table = Table(wrapped_rows, colWidths=[42, 165, 245], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#4A5F73")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F6F")),
                    ("BACKGROUND", (0, 1), (-1, 2), colors.HexColor("#EAF0F5")),
                    ("ROWBACKGROUNDS", (0, 3), (-1, -1), [colors.white, colors.HexColor("#F8FBFD")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        elements.append(table)
        if i == len(product_items) - 1:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Queries</b>", section_title_style))
            if queries:
                for idx, query in enumerate(queries, start=1):
                    elements.append(Paragraph(f"{idx}. {query}", section_body_style))
            else:
                # Keep visible space reserved for queries when none are provided.
                elements.append(Spacer(1, 28))

    elements.append(PageBreak())

    analysis_lines = _build_project_requirement_analysis_lines(product_items, customer_name)
    analysis_headings = {
        "Customer",
        "Current Status",
        "Migration Requirement",
        "Technical Requirements",
        "Migration Reason",
        "Shortlisted Chipsets",
        "Memory Requirement",
        "Technical Observations & Considerations",
    }
    for line in analysis_lines:
        text = line.strip()
        if not text:
            elements.append(Spacer(1, 4))
            continue

        if text == "Project Requirement Analysis":
            elements.append(Paragraph(f"<b>{text}</b>", section_title_style))
        elif text in analysis_headings or text.startswith("Product "):
            elements.append(Paragraph(f"<b>{text}</b>", section_heading_style))
        else:
            elements.append(Paragraph(text, section_body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
