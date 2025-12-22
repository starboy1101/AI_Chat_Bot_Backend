import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

PDF_FIELD_MAPPING = {
    "start": "Type of service",
    "Optimization_type": "Type of Optimization",
    "App_type": "App Type",
    "DSP_Processor": "Type of processor",
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
    "TargetPlatform_2": "Chipset Details"
}


def build_pdf_rows(context: dict) -> list[list[str]]:
    pcm_size = context.get("Audio_Params_1", "NA")

    rows = [
        ["REQ ID", "RQ01"],
        ["Purpose", f"Audio Output should support {pcm_size} sample"],
    ]

    for key, label in PDF_FIELD_MAPPING.items():
        # ✅ Only include if key exists in context
        if key not in context:
            continue

        value = context.get(key)
        rows.append([label, value if value else "NA"])

    rows.extend([
        ["Derived Requirement", "NA"],
        ["Requirement Priority", "Urgent"],
        ["Access Restrictions", "NA"],
        ["Input(s)", "Audio Input Sample"],
        ["Output(s)", "Audio Output Sample"],
        ["Process", "NA"],
        ["Mandatory Fields", "NA"],
        ["Pre-Loaded Values", "Customer provided"],
        ["Default Values", "16 bit"],
        ["Valid range of Values", "[8, 16, 24, 32]"],
        ["Data Latency Period", "NA"],
        ["Data Retention Period", "NA"],
    ])

    return rows

def build_queries(context: dict) -> list[str]:
    queries = context.get("query_input")

    if not queries:
        return []

    # If single string → convert to list
    if isinstance(queries, str):
        return [queries]

    # If already list
    if isinstance(queries, list):
        return queries

    return []


def generate_final_requirements_pdf(context: dict) -> bytes:
    print("PDF CONTEXT KEYS:", context.keys())

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Requirements Specification</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    table = Table(build_pdf_rows(context), colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(table)

    queries = build_queries(context)

    if queries:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Your Queries</b>", styles["Heading2"]))

        for idx, q in enumerate(queries, start=1):
            elements.append(Paragraph(f"{idx}. {q}", styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)
    return buffer.read()

