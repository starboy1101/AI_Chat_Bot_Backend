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


REQUIREMENT_FIELD_LABELS: Dict[str, str] = {
    "purpose": "Purpose1",
    "inputs": "Input(s)",
    "outputs": "Output(s)",
    "process": "Process",
    "validation": "Validation Rules/ Verification criteria",
    "acceptance_criteria": "Acceptance Criteria",
    "testability": "Testability with respect to test environment",
    "feasible": "Feasible within project constraints",
    "critical": "Critical",
    "derived_requirement": "Derived Requirement",
    "access_restrictions": "Access Restrictions",
    "mandatory_fields": "Mandatory Fields",
    "pre_loaded_values": "Pre-Loaded Values",
    "default_values": "Default Values",
    "valid_range_of_values": "Valid range of Values",
    "data_latency_period": "Data Latency Period",
    "data_retention_period": "Data Retention Period",
    "data_rate": "Data Rate/ Daily Number of transaction",
    "external_events": "External Events",
    "temporal_events": "Temporal Events",
    "constraints": "Constraints",
    "effects_on_other_systems": "Effects on other systems/sub system",
    "assumptions": "Assumptions",
    "failure_scenario": "Failure Scenario",
    "action_on_failure": "Action if Failure",
    "status": "Requirement Status",
    "comments": "Comments",
    "asil": "ASIL Level",
}


REQUIREMENT_FIELD_DEFINITIONS: Dict[str, str] = {
    "purpose": (
        "The exact behavior or functionality the system must provide for this requirement. "
        "Prefer the highest-level shall/must/should statement over examples, list items, timing tables, "
        "metadata, validation text, or debug-only details."
    ),
    "inputs": "Signals, data, user actions, documents, states, or external events needed to execute the requirement.",
    "outputs": "Observable results, stored data, messages, reports, state changes, or responses produced by the requirement.",
    "process": (
        "The step-by-step logic, workflow, state transition, or processing sequence for the requirement. "
        "Do not include derived-requirement statements, priority, access restrictions, validation, acceptance criteria, or other metadata."
    ),
    "validation": (
        "How the requirement will be verified, including rules, checks, tools, test method, and pass/fail criteria. "
        "Only include verification/test evidence; do not include purpose, description, inputs, outputs, or nearby shall statements."
    ),
    "acceptance_criteria": (
        "The conditions that must be satisfied before the requirement is accepted by the customer or project team. "
        "Do not copy validation criteria verbatim; express the observable acceptance condition for the requirement."
    ),
    "testability": "Whether this requirement can be tested in the planned environment, and in which test phase or setup.",
    "feasible": "Whether the requirement can be implemented within project constraints such as cost, schedule, tools, hardware, and operating environment.",
    "critical": "Whether failure to implement this requirement creates technical, customer, cost, competence, schedule, safety, or quality risk.",
    "derived_requirement": "The parent, source, safety, system, or customer requirement from which this requirement is derived, if any.",
    "access_restrictions": "Roles, permissions, authentication, or authorization constraints that limit who can use this functionality.",
    "mandatory_fields": "Fields or data elements that must be present before the requirement can be processed.",
    "pre_loaded_values": "Values already maintained by the client, device, database, or environment before processing starts.",
    "default_values": "Values used automatically when no explicit value is supplied.",
    "valid_range_of_values": "Allowed value range, enum, boundary, or format for relevant inputs or signals.",
    "data_latency_period": "Maximum acceptable request, response, signal, or processing time.",
    "data_retention_period": "How long related data must be stored or maintained.",
    "data_rate": "Expected transaction, signal, memory transfer, frame, or network rate.",
    "external_events": "External triggers that start this functionality.",
    "temporal_events": "Time-based triggers, schedules, delays, or periodic events related to this functionality.",
    "constraints": "Implementation, API, design, hardware, software, or environmental limits that affect the requirement.",
    "effects_on_other_systems": "Known impact on other features, systems, subsystems, users, interfaces, or data.",
    "assumptions": "Conditions assumed to be true while defining this requirement.",
    "failure_scenario": "What can go wrong while this requirement executes.",
    "action_on_failure": "Recovery, fallback, alert, retry, or mitigation behavior when failure occurs.",
    "status": "Lifecycle state of the requirement, such as proposed, accepted, reviewed, delivered, or verified.",
    "comments": "Additional rationale, notes, evidence, or reviewer comments.",
    "asil": "Automotive Safety Integrity Level classification for a safety requirement.",
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


def build_requirement_field_question(field: str, req_id: str | None = None) -> str:
    label = REQUIREMENT_FIELD_LABELS.get(field, field.replace("_", " ").title())
    definition = REQUIREMENT_FIELD_DEFINITIONS.get(field)
    subject = f" for {req_id}" if req_id else ""
    question = f"Please provide {label}{subject}."
    if definition:
        question = f"{question}\n\nDefinition:\n{definition}"
    return question
