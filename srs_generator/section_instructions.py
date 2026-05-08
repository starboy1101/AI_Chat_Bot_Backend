from __future__ import annotations

from typing import Any, Dict


PROJECT_SECTION_INSTRUCTIONS: Dict[str, str] = {
    "purpose": "Describe the purpose of this Software Requirements Specification and its usage.",
    "intended_audience": (
        "Describe the different types of readers that the document is intended for, "
        "such as developers, project team members, marketing staff, users, testers, etc."
    ),
    "system_overview": (
        "Provide a description of the environment where the project being specified resides. "
        "Briefly explain all the relevant entities and their interactions.\n\n"
        "Provide details of existing software, if any, for its problems and need for the proposed "
        "software/product.\n\n"
        "Provide description of the software/product being specified and its purpose, its boundaries, "
        "its relation with other entities, relevant benefits, objectives, and goals."
    ),
    "scope": (
        "Define scope of the software as addressed by this SRS document in terms of the software block "
        "diagram, features to be included and excluded.\n\n"
        "The software block diagram is a logical view of the software being specified with a set of "
        "logical blocks and each block has a set of requirements. Briefly explain the identified block. "
        "Its requirements must be documented in the subsequent section.\n\n"
        "For automotive projects, include the security concept architecture if applicable."
    ),
    "operating_environment": (
        "Analyze if any requirements have an impact on the operating environment. Include third-party "
        "software, hardware interfaces such as signals, signal quality, voltage and current, "
        "environmental conditions such as temperature and EMC, performance such as response time and "
        "processing time, and hardware resource constraints.\n\n"
        "Define the operating environment and the other elements interacting with the real-time software."
    ),
    "acceptance_criteria": (
        "Clearly state the acceptance criteria of the product/project as agreed with the customer. "
        "If the project/product will be done in phases, include the acceptance criteria for each phase."
    ),
}


PROJECT_FIELD_LABELS: Dict[str, str] = {
    "purpose": "Purpose",
    "intended_audience": "Intended audience",
    "system_overview": "System overview",
    "scope": "Scope",
    "operating_environment": "Operating environment",
    "acceptance_criteria": "Acceptance criteria",
}


def build_project_field_question(field: str, project: Any | None = None) -> str:
    label = PROJECT_FIELD_LABELS.get(field, field.replace("_", " ").title())
    instruction = PROJECT_SECTION_INSTRUCTIONS.get(field)
    question = f"Please provide the SRS {label}."
    if instruction:
        question = f"{question}\n\nTemplate instruction:\n{instruction}"

    if field == "scope" and project is not None:
        overview = getattr(project, "system_overview", "")
        if overview and overview != "Not specified.":
            question = f"{question}\n\nExtracted overview for context:\n{overview[:1200]}"

    return question

