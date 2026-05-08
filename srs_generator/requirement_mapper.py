from __future__ import annotations

from typing import Any, Dict, List

from srs_generator.models import RequirementMetadata, SRSProject, StructuredRequirement


LEGACY_LABEL_TO_FIELD = {
    "service_select": "purpose",
    "Optimization_type": "process",
    "App_type": "purpose",
    "Porting_question_1": "inputs",
    "Porting_question_2": "validation",
    "DSP_Processor": "outputs",
    "Application": "purpose",
    "Audio_Interface": "interfaces",
    "Audio_Params_1": "inputs",
    "Audio_Params_2": "inputs",
    "Audio_Params_3": "outputs",
    "Audio_Tech_1": "process",
    "Audio_Tech_2": "inputs",
    "CodeBase_1": "inputs",
    "CodeBase_2": "inputs",
    "CodeBase_3": "constraints",
    "CodeBase_4": "constraints",
    "CodeBase_5": "constraints",
    "CodeBase_6": "process",
    "TargetPlatform_1": "outputs",
    "TargetPlatform_2": "outputs",
}


def legacy_context_to_srs_project(context: Dict[str, Any], project_name: str = "Generated SRS") -> SRSProject:
    """Convert the old flat questionnaire context into the new requirement-centric model."""
    product_contexts = _normalize_product_contexts(context)
    requirements: List[StructuredRequirement] = []

    for product_name, product_ctx in product_contexts.items():
        fields: Dict[str, List[str]] = {}
        for qid, entry in product_ctx.items():
            target = LEGACY_LABEL_TO_FIELD.get(qid)
            if not target:
                continue
            value = _entry_text(entry)
            if value == "N/A":
                continue
            fields.setdefault(target, []).append(value)

        requirements.append(
            StructuredRequirement(
                logical_block=product_name if product_name != "default" else "General",
                req_id=f"AUTO_REQ_{len(requirements) + 1:03d}",
                requirement_type="functional",
                purpose=_join(fields.get("purpose")) or f"Capture implementation requirement for {product_name}.",
                inputs=_join(fields.get("inputs")) or "Not specified.",
                outputs=_join(fields.get("outputs")) or "Not specified.",
                process=_join(fields.get("process")) or "Not specified.",
                validation=_join(fields.get("validation")) or "Validate against agreed project acceptance criteria.",
                acceptance_criteria="Requirement is accepted when implementation evidence and validation results meet the captured constraints.",
                constraints=_join(fields.get("constraints")) or "Not specified.",
                metadata=RequirementMetadata(
                    source_section=product_name,
                    extraction_method="legacy_context",
                    confidence=0.75,
                ),
            )
        )

    return SRSProject(
        project_name=project_name,
        requirements=requirements,
        source_name="chat_context",
        system_overview="Generated from captured chat requirements.",
        scope="Software requirements captured through the guided assistant flow.",
    )


def _normalize_product_contexts(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(context, dict):
        return {"default": {}}
    if any(key in LEGACY_LABEL_TO_FIELD for key in context):
        return {"default": context}
    result = {str(name): value for name, value in context.items() if isinstance(value, dict)}
    return result or {"default": {}}


def _entry_text(entry: Any) -> str:
    if not isinstance(entry, dict):
        return "N/A"
    value = entry.get("value")
    if value in (None, "", []):
        return "N/A"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _join(values: List[str] | None) -> str:
    if not values:
        return ""
    return "\n".join(f"- {value}" for value in values if value)

