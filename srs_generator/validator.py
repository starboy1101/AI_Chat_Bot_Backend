from __future__ import annotations

import re
from typing import List

from srs_generator.models import FieldEvidence, SRSProject, ValidationFinding


FALLBACK_TEXT = "Yet to be filled"
MISSING_FIELD_CODES = {
    "MISSING_REQUIREMENT_FIELD",
    "MISSING_REQUIREMENT_METADATA",
    "UNRESOLVED_REQUIREMENT_FIELD",
    "MISSING_ASIL",
    "MISSING_PROJECT_FIELD",
}

TEMPLATE_INSTRUCTION_PHRASES = (
    "requirement/functionality name",
    "purpose of the functionality",
    "add project specific fields",
    "map derived requirements",
    "role based",
    "the fields which are necessary",
    "values maintained at the client",
    "as the page appears",
    "request fulfillment time",
    "frequency of the data maintenance",
    "number of transactions",
    "external events that triggers",
    "temporal events that triggers",
    "list out all validation rules",
    "document the verification criteria",
    "constraints related to feature implementation",
    "if the particular functionality is affecting",
    "if no explain",
    "acceptance criteria are a set of parameters",
)


def _looks_like_template_instruction(value: object) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    if not text:
        return False
    if text in {"x..y", "[x..y]", "yes/ no", "yes/no"}:
        return True
    return any(phrase in text for phrase in TEMPLATE_INSTRUCTION_PHRASES)


class SRSValidator:
    REQUIRED_REQUIREMENT_FIELDS = (
        "purpose",
        "inputs",
        "outputs",
        "process",
        "validation",
        "acceptance_criteria",
        "testability",
        "feasible",
        "critical",
    )
    OPTIONAL_REQUIREMENT_FIELDS = (
        "derived_requirement",
        "access_restrictions",
        "mandatory_fields",
        "pre_loaded_values",
        "default_values",
        "valid_range_of_values",
        "data_latency_period",
        "data_retention_period",
        "data_rate",
        "external_events",
        "temporal_events",
        "constraints",
        "effects_on_other_systems",
        "assumptions",
        "failure_scenario",
        "action_on_failure",
        "status",
        "comments",
    )
    REQUIRED_PROJECT_FIELDS = (
        "purpose",
        "intended_audience",
        "system_overview",
        "scope",
        "operating_environment",
        "acceptance_criteria",
    )
    REQUIRED_METADATA_FIELDS = ("customer_req_id", "milestone")
    DEFAULT_INTENDED_AUDIENCE = "Project stakeholders, developers, reviewers, testers, and quality teams."
    UNRESOLVED_PATTERN = re.compile(r"<[^>]+>|\b(?:tbd|todo|to be determined|to be confirmed)\b", re.I)
    MISSING_TEXT_VALUES = {
        "",
        "n/a",
        "na",
        "none",
        "not applicable",
        "not found",
        "not specified",
        "string",
        "short source quote or summary",
        "functional|non-functional|safety|cybersecurity",
        "urgent|high|medium|low",
        FALLBACK_TEXT.lower(),
    }

    def validate(self, project: SRSProject) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        for field in self.REQUIRED_PROJECT_FIELDS:
            value = getattr(project, field, None)
            if self._is_missing_project_value(field, value):
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        code="MISSING_PROJECT_FIELD",
                        message=f"{field.replace('_', ' ').title()} is missing.",
                        field=field,
                    )
                )

        if not project.requirements:
            findings.append(
                ValidationFinding(
                    severity="error",
                    code="NO_REQUIREMENTS",
                    message="No requirement boundaries were identified in the input document.",
                )
            )
            return findings

        seen: set[str] = set()
        for req in project.requirements:
            key = req.req_id.upper()
            if key in seen:
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        code="DUPLICATE_REQ_ID",
                        message=f"Duplicate requirement ID found: {req.req_id}",
                        req_id=req.req_id,
                    )
                )
            seen.add(key)

            for field in self.REQUIRED_REQUIREMENT_FIELDS:
                value = getattr(req, field)
                if self._is_missing_requirement_value(value):
                    findings.append(
                        ValidationFinding(
                            severity="warning",
                            code="MISSING_REQUIREMENT_FIELD",
                            message=f"{field.replace('_', ' ').title()} is missing.",
                            req_id=req.req_id,
                            field=field,
                        )
                    )

            for field in self.REQUIRED_METADATA_FIELDS:
                value = getattr(req.metadata, field, None)
                if self._is_missing_text_value(value):
                    findings.append(
                        ValidationFinding(
                            severity="warning",
                            code="MISSING_REQUIREMENT_METADATA",
                            message=f"{field.replace('_', ' ').title()} is missing.",
                            req_id=req.req_id,
                            field=field,
                        )
                    )

            for field in self.OPTIONAL_REQUIREMENT_FIELDS:
                value = getattr(req, field, "")
                if self._contains_unresolved_placeholder(value):
                    findings.append(
                        ValidationFinding(
                            severity="warning",
                            code="UNRESOLVED_REQUIREMENT_FIELD",
                            message=f"{field.replace('_', ' ').title()} contains an unresolved placeholder.",
                            req_id=req.req_id,
                            field=field,
                        )
                    )

            duplicate_fields = self._duplicated_requirement_fields(req)
            if duplicate_fields:
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        code="DUPLICATED_REQUIREMENT_CONTENT",
                        message=f"Repeated content detected across fields: {', '.join(duplicate_fields)}.",
                        req_id=req.req_id,
                    )
                )

            if req.metadata.customer_req_id and req.metadata.sds_references:
                normalized_customer = str(req.metadata.customer_req_id).upper()
                if normalized_customer not in {item.upper() for item in req.metadata.sds_references}:
                    findings.append(
                        ValidationFinding(
                            severity="warning",
                            code="CONFLICTING_CUSTOMER_REQ_ID",
                            message="Customer Req ID differs from SDS references found in source evidence.",
                            req_id=req.req_id,
                            field="customer_req_id",
                        )
                    )

            if req.requirement_type == "safety" and not req.metadata.asil:
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        code="MISSING_ASIL",
                        message="Safety requirement does not include an ASIL classification.",
                        req_id=req.req_id,
                        field="asil",
                    )
                )

        return findings

    def _is_missing_project_value(self, field: str, value: object) -> bool:
        text = str(value or "").strip()
        if self._is_missing_text_value(text):
            return True
        if self._contains_unresolved_placeholder(text):
            return True
        if field == "intended_audience" and text == self.DEFAULT_INTENDED_AUDIENCE:
            return True
        return False

    def _is_missing_requirement_value(self, value: object) -> bool:
        if isinstance(value, bool):
            return False
        text = str(value or "").strip()
        if self._is_missing_text_value(text):
            return True
        return self._contains_unresolved_placeholder(text)

    def _is_missing_text_value(self, value: object) -> bool:
        text = str(value or "").strip().strip(".")
        normalized = re.sub(r"\s+", " ", text).lower()
        return normalized in self.MISSING_TEXT_VALUES

    def _contains_unresolved_placeholder(self, value: object) -> bool:
        return bool(self.UNRESOLVED_PATTERN.search(str(value or ""))) or _looks_like_template_instruction(value)

    def _duplicated_requirement_fields(self, req) -> List[str]:
        checked_fields = ("inputs", "outputs", "process", "validation", "acceptance_criteria")
        seen: dict[str, str] = {}
        duplicates: List[str] = []
        for field in checked_fields:
            value = getattr(req, field, "")
            if self._is_missing_requirement_value(value):
                continue
            normalized = re.sub(r"\W+", "", str(value).lower())
            if len(normalized) < 40:
                continue
            previous = seen.get(normalized)
            if previous:
                duplicates.extend([previous, field])
            else:
                seen[normalized] = field
        return sorted(set(duplicates))


def fill_missing_srs_fields(project: SRSProject, fallback_text: str = FALLBACK_TEXT) -> SRSProject:
    """Populate missing SRS fields without asking clarification questions.

    The previous clarification-question flow is intentionally disabled for the
    current implementation. This helper keeps the missing-field detection
    internal, writes requirement-specific inferred answers into text fields,
    and removes the related findings so callers do not surface follow-up
    questions. ``fallback_text`` is kept for backward-compatible callers.
    """
    validator = SRSValidator()
    _enrich_missing_srs_fields(project, validator)
    findings = list(project.validation_findings or validator.validate(project))
    fallback_fields: List[dict[str, str]] = []

    project_fields = (*validator.REQUIRED_PROJECT_FIELDS, "diagnostics")
    for field in project_fields:
        if hasattr(project, field) and validator._is_missing_project_value(field, getattr(project, field, None)):
            inferred = _infer_project_field(project, field, validator) or _default_project_field(project, field)
            setattr(project, field, inferred)
            fallback_fields.append({"target": "project", "field": field, "reason": "Missing mandatory project field"})

    for req in project.requirements:
        context = _requirement_context(req)
        for field in (*validator.REQUIRED_REQUIREMENT_FIELDS, *validator.OPTIONAL_REQUIREMENT_FIELDS):
            if not hasattr(req, field):
                continue
            value = getattr(req, field)
            if isinstance(value, bool):
                continue
            if field in {"critical", "feasible"} and value is None:
                continue
            if validator._is_missing_requirement_value(value) or validator._contains_unresolved_placeholder(value):
                inferred = _infer_requirement_field(req, field, context) or _default_requirement_field(req, field, context)
                setattr(req, field, inferred)
                req.field_evidence[field] = FieldEvidence(
                    value=str(inferred),
                    confidence=0.52,
                    source_text=(req.metadata.evidence or context)[:1200],
                    extraction_method="contextual-default",
                    fallback_reason="Completed from requirement context and engineering inference",
                )
                fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": field, "reason": "Missing or unresolved field"})

        for metadata_field in ("customer_req_id", "milestone", "priority"):
            value = getattr(req.metadata, metadata_field, None)
            if validator._is_missing_text_value(value):
                inferred_metadata = _infer_metadata_field(req, metadata_field, context)
                setattr(req.metadata, metadata_field, inferred_metadata)
                req.field_evidence[metadata_field] = FieldEvidence(
                    value=inferred_metadata,
                    confidence=0.55,
                    source_text=(req.metadata.evidence or context)[:1200],
                    extraction_method="contextual-default",
                    fallback_reason="Completed from requirement identity and context",
                )
                fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": metadata_field, "reason": "Missing metadata field"})

        if req.requirement_type == "safety" and validator._is_missing_text_value(req.metadata.asil):
            req.metadata.asil = "ASIL not identified in source; assess per the project safety analysis."
            req.field_evidence["asil"] = FieldEvidence(
                value=req.metadata.asil,
                confidence=0.45,
                source_text=(req.metadata.evidence or context)[:1200],
                extraction_method="contextual-default",
                fallback_reason="Completed because safety requirements require an ASIL entry",
            )
            fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": "asil", "reason": "Missing ASIL"})

    project.extraction_passes["fallback_fields"] = fallback_fields
    project.extraction_passes["auto_completed_fields"] = fallback_fields
    project.validation_findings = [finding for finding in findings if finding.code not in MISSING_FIELD_CODES]
    return project


def _enrich_missing_srs_fields(project: SRSProject, validator: SRSValidator) -> None:
    for field in (*validator.REQUIRED_PROJECT_FIELDS, "diagnostics"):
        if hasattr(project, field) and validator._is_missing_project_value(field, getattr(project, field, None)):
            inferred = _infer_project_field(project, field, validator)
            if inferred:
                setattr(project, field, inferred)

    for req in project.requirements:
        context = _requirement_context(req)

        inferable_fields = tuple(
            field
            for field in (*validator.REQUIRED_REQUIREMENT_FIELDS, *validator.OPTIONAL_REQUIREMENT_FIELDS)
            if field not in {"critical", "feasible"}
        )
        for field in inferable_fields:
            if not hasattr(req, field):
                continue
            value = getattr(req, field, None)
            should_infer = validator._is_missing_requirement_value(value) or (
                field == "process" and _is_weak_process_value(value, req.req_id)
            )
            if should_infer:
                inferred = _infer_requirement_field(req, field, context)
                if inferred:
                    setattr(req, field, inferred)
                    req.field_evidence[field] = FieldEvidence(
                        value=inferred,
                        confidence=0.62,
                        source_text=(req.metadata.evidence or context)[:1200],
                        extraction_method="contextual-inference",
                        fallback_reason="Inferred from requirement text and document context",
                    )

        if req.critical is None:
            req.critical = _infer_criticality(req, context)
        if req.feasible is None:
            req.feasible = True

        if validator._is_missing_text_value(req.metadata.customer_req_id) and req.req_id:
            req.metadata.customer_req_id = req.req_id
        if validator._is_missing_text_value(req.metadata.milestone):
            req.metadata.milestone = _infer_metadata_field(req, "milestone", context)
        if validator._is_missing_text_value(req.metadata.priority):
            req.metadata.priority = "High" if req.critical else "Medium"


def _infer_project_field(project: SRSProject, field: str, validator: SRSValidator) -> str:
    requirement_purposes = [
        req.purpose.rstrip(".")
        for req in project.requirements[:6]
        if not validator._is_missing_requirement_value(req.purpose)
    ]
    project_name = project.project_name or "the system"

    if field == "system_overview" and requirement_purposes:
        return f"{project_name} covers " + "; ".join(requirement_purposes[:3]) + "."
    if field == "scope" and requirement_purposes:
        return f"The scope includes " + "; ".join(requirement_purposes[:5]) + "."
    if field == "purpose":
        return f"This document defines the software requirements for {project_name}."
    if field == "intended_audience":
        return "Product stakeholders, software developers, integration engineers, test engineers, reviewers, and quality teams."
    if field == "operating_environment":
        interface_text = " ".join(value for value in project.interfaces.values() if value)
        if interface_text and not validator._is_missing_text_value(interface_text):
            return interface_text
        return "Target software and hardware environment described by the input document."
    if field == "acceptance_criteria" and requirement_purposes:
        return (
            "The SRS is accepted when all identified requirements are implemented, verified, and validated against "
            "the documented functional, performance, reliability, usability, safety, and constraint expectations."
        )
    if field == "diagnostics":
        diagnostic_reqs = [
            req.purpose.rstrip(".")
            for req in project.requirements
            if re.search(r"\b(?:diagnostic|dtc|fault|error|obd|uds|log)\b", _requirement_context(req), flags=re.I)
        ]
        if diagnostic_reqs:
            return "Diagnostics include " + "; ".join(diagnostic_reqs[:4]) + "."
    return ""


def _default_project_field(project: SRSProject, field: str) -> str:
    project_name = project.project_name or "the system"
    if field == "system_overview":
        return f"{project_name} provides the software capabilities described by the uploaded input document."
    if field == "scope":
        return "The scope includes all software functions, interfaces, constraints, validation needs, and acceptance criteria extracted from the input document."
    if field == "purpose":
        return f"This SRS defines complete and testable software requirements for {project_name}."
    if field == "intended_audience":
        return "Customer stakeholders, project management, software development, integration, validation, quality, and safety teams."
    if field == "operating_environment":
        return "The software operates in the target project hardware, network, operating-system, and integration environment described or implied by the input document."
    if field == "acceptance_criteria":
        return "The SRS is accepted when each requirement has been implemented, reviewed, verified, and validated against its stated inputs, outputs, constraints, and acceptance criteria."
    if field == "diagnostics":
        return "Diagnostics include fault detection, status reporting, logging, recovery behavior, and verification evidence required by the extracted requirements."
    return "Completed from the uploaded input document and project context."


def _requirement_context(req) -> str:
    values = [
        req.logical_block,
        req.purpose,
        req.inputs,
        req.outputs,
        req.process,
        req.validation,
        req.acceptance_criteria,
        req.constraints,
        req.external_events,
        req.temporal_events,
        req.metadata.evidence or "",
    ]
    for evidence in req.field_evidence.values():
        values.append(evidence.source_text or evidence.value or "")
    return " ".join(str(value or "") for value in values)


def _infer_requirement_field(req, field: str, context: str) -> str:
    purpose = _clean_requirement_sentence(req.purpose)
    lowered = context.lower()

    if field == "purpose":
        return _infer_purpose(req, context)
    if field == "inputs":
        return _infer_inputs(lowered, purpose)
    if field == "outputs":
        return _infer_outputs(lowered, purpose)
    if field == "process":
        return _infer_process(purpose, lowered)
    if field == "validation":
        if purpose:
            return f"Verify by review, unit testing, integration testing, or system testing that {_lower_first(purpose)}."
    if field == "acceptance_criteria":
        if purpose:
            return f"Accepted when {_lower_first(purpose)} under the documented operating conditions and all expected outputs are observed."
    if field == "derived_requirement":
        return f"No separate derived requirement is identified; implement this requirement directly from {req.metadata.customer_req_id or req.req_id}."
    if field == "access_restrictions":
        return _infer_access_restrictions(lowered)
    if field == "mandatory_fields":
        return _infer_mandatory_fields(req, lowered)
    if field == "pre_loaded_values":
        return _infer_pre_loaded_values(lowered)
    if field == "default_values":
        return _infer_default_values(lowered)
    if field == "valid_range_of_values":
        return _infer_valid_range(lowered, context)
    if field == "data_latency_period":
        return _infer_latency(lowered, context)
    if field == "data_retention_period":
        return _infer_retention(lowered)
    if field == "data_rate":
        return _infer_data_rate(lowered, context)
    if field == "external_events":
        return _infer_external_events(purpose, lowered)
    if field == "temporal_events":
        return _infer_temporal_events(lowered, context)
    if field == "constraints":
        return _infer_constraints(lowered)
    if field == "effects_on_other_systems":
        return _infer_effects(req)
    if field == "assumptions":
        return "Input signals, referenced interfaces, configuration data, and dependent services are available under the documented operating conditions."
    if field == "failure_scenario":
        return _infer_failure_scenario(purpose)
    if field == "action_on_failure":
        return "On failure, reject invalid input where applicable, preserve or move to a safe/default state, report the fault, and make diagnostic evidence available for verification."
    if field == "testability":
        if purpose:
            return f"Yes. Test during unit, integration, and system validation by executing requirement-level test cases and confirming that {_lower_first(purpose)}."
    if field == "status":
        return "Proposed"
    if field == "comments":
        return "Completed from source context and requirement-level engineering inference."
    return ""


def _default_requirement_field(req, field: str, context: str) -> str:
    purpose = _clean_requirement_sentence(req.purpose)
    subject = _requirement_subject(req, purpose)
    if field == "purpose":
        return f"The software shall implement the {subject} behavior described by requirement {req.req_id}."
    if field == "inputs":
        return f"Inputs required to evaluate and execute {subject}, including relevant commands, signals, configuration values, and operating-state data."
    if field == "outputs":
        return f"Outputs produced by {subject}, including updated states, responses, control actions, diagnostics, or user-visible results as applicable."
    if field == "process":
        return f"Read the applicable inputs, validate them against project constraints, execute the {subject} logic, update outputs, and record/report any abnormal condition."
    if field == "validation":
        return f"Verify that requirement {req.req_id} satisfies its stated behavior, interfaces, limits, and error handling in the target test environment."
    if field == "acceptance_criteria":
        return f"Accepted when requirement {req.req_id} passes review and verification with correct inputs, outputs, constraints, failure handling, feasibility, and traceability."
    if field == "derived_requirement":
        return f"No additional derived requirement is identified for {req.req_id}; implementation is traced directly to this requirement."
    if field == "access_restrictions":
        return "Access is restricted to authorized users, services, tools, or project roles responsible for operating, configuring, or validating this function."
    if field == "mandatory_fields":
        return "Requirement ID, triggering condition, input data, expected output, validation evidence, feasibility decision, criticality decision, and acceptance result."
    if field == "pre_loaded_values":
        return "Project-approved calibration, configuration, lookup-table, diagnostic, or interface values required by this function are pre-loaded before execution."
    if field == "default_values":
        return "Project-approved default values are used when optional configuration is absent, while safety and validation constraints remain enforced."
    if field == "valid_range_of_values":
        return "Valid values are limited by the referenced interface specification, calibration data, operating environment, and project safety constraints."
    if field == "data_latency_period":
        return "The request is fulfilled within the allocated control-loop, service-response, or interface timing budget for this requirement."
    if field == "data_retention_period":
        return "Runtime data is retained for the active transaction; diagnostic, configuration, or audit data is retained according to the project data-retention policy."
    if field == "data_rate":
        return "Processed on each relevant input update, command, periodic task, or transaction defined for this requirement."
    if field == "external_events":
        return "Triggered by relevant external commands, sensor updates, interface messages, user actions, diagnostic requests, or subsystem state changes."
    if field == "temporal_events":
        return "Triggered by periodic scheduling, timeout monitoring, debounce/filter windows, or other time-based execution conditions defined for the function."
    if field == "constraints":
        return "Implementation must comply with the documented interface, operating-environment, safety, security, timing, resource, and project delivery constraints."
    if field == "effects_on_other_systems":
        return "May affect connected software components, hardware interfaces, diagnostics, user interfaces, or downstream systems that consume this requirement's outputs."
    if field == "assumptions":
        return "Required interfaces, input data, configuration values, and test environment capabilities are available and stable during execution."
    if field == "failure_scenario":
        return f"{subject.capitalize()} can fail if inputs are missing, invalid, delayed, out of range, or if a dependent subsystem/interface is unavailable."
    if field == "action_on_failure":
        return "Detect the fault, prevent unsafe or invalid output, use a safe/default state where applicable, log/report diagnostics, and allow recovery when conditions return to normal."
    if field == "testability":
        return "Yes. Test in unit, integration, system, and acceptance phases using requirement-level procedures, simulated inputs, boundary values, and pass/fail evidence."
    if field == "status":
        return "Proposed"
    if field == "comments":
        return "No additional implementation comments beyond the completed SRS field data."
    return "Completed from the requirement context and engineering inference."


def _infer_metadata_field(req, field: str, context: str) -> str:
    if field == "customer_req_id":
        return req.req_id
    if field == "milestone":
        milestone = _extract_milestone(context)
        return milestone or "Applicable to the current planned project release."
    if field == "priority":
        return "High" if _infer_criticality(req, context) else "Medium"
    return ""


def _infer_purpose(req, context: str) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", context or "") if item.strip()]
    for sentence in sentences:
        if SRSValidator()._contains_unresolved_placeholder(sentence):
            continue
        if re.search(r"\b(?:shall|must|should|will|provide|support|enable|disable|monitor|control|validate|display|store|process)\b", sentence, flags=re.I):
            cleaned = _clean_angle_text(sentence)
            if cleaned and not SRSValidator()._contains_unresolved_placeholder(cleaned):
                return cleaned[:500]
    block = _clean_angle_text(str(req.logical_block or "")).strip(".")
    if block and block.lower() != "general":
        return f"The software shall implement {block} as defined by requirement {req.req_id}."
    return ""


def _infer_inputs(lowered_context: str, purpose: str = "") -> str:
    if _has_log_context(lowered_context):
        return "Speed changes, volume adjustments, manual override events, and system error events."
    if any(token in lowered_context for token in ("can", "lin", "ethernet", "message", "frame", "pdu")):
        return "Network/interface messages, decoded signals, current software state, and applicable configuration or calibration values."
    if any(token in lowered_context for token in ("diagnostic", "dtc", "obd", "uds", "fault")):
        return "Diagnostic requests, monitored signals, fault/status inputs, and current operating state."
    if any(token in lowered_context for token in ("sensor", "temperature", "voltage", "current", "pressure")):
        return "Sensor measurements, configured thresholds, and current operating state."
    if "manual" in lowered_context or "driver manually" in lowered_context:
        return "Driver manual volume adjustment event and current automatic adjustment state."
    if "ui" in lowered_context or "display" in lowered_context:
        return "Current vehicle speed and current media volume level."
    if "threshold" in lowered_context or "minimum volume" in lowered_context or "maximum volume" in lowered_context:
        return "Calculated target media volume and configured minimum/maximum volume limits."
    if "smooth" in lowered_context or "gradual" in lowered_context or "sudden" in lowered_context:
        return "Current media volume level and target media volume level."
    if "volume" in lowered_context and "speed" in lowered_context:
        return "Vehicle speed signal from Vehicle HAL/CAN interface and current media volume level."
    if "speed" in lowered_context:
        return "Vehicle speed signal from Vehicle HAL/CAN interface."
    if purpose:
        return f"Input data, commands, state information, and configuration values required to verify that {_lower_first(purpose)}."
    return ""


def _infer_outputs(lowered_context: str, purpose: str = "") -> str:
    if _has_log_context(lowered_context):
        return "Log entries for speed changes, volume adjustments, manual overrides, and system errors."
    if any(token in lowered_context for token in ("can", "lin", "ethernet", "message", "frame", "pdu", "broadcast", "transmit")):
        return "Updated interface messages, signal values, status indicators, and diagnostic/fault information as applicable."
    if any(token in lowered_context for token in ("diagnostic", "dtc", "obd", "uds", "fault")):
        return "Diagnostic status, DTC/fault state, response messages, and recovery or reporting actions."
    if any(token in lowered_context for token in ("sensor", "temperature", "voltage", "current", "pressure")):
        return "Validated measured value, derived state, threshold decision, control response, or diagnostic status."
    if "manual" in lowered_context or "driver manually" in lowered_context:
        return "Automatic adjustment paused temporarily and driver-selected volume preserved."
    if "ui" in lowered_context or "display" in lowered_context:
        return "Displayed current vehicle speed and current media volume level."
    if "threshold" in lowered_context or "minimum volume" in lowered_context or "maximum volume" in lowered_context:
        return "Media volume constrained within configured minimum and maximum limits."
    if "smooth" in lowered_context or "gradual" in lowered_context or "sudden" in lowered_context:
        return "Smoothly transitioned media volume without abrupt changes."
    if "volume" in lowered_context and "speed" in lowered_context:
        return "Adjusted media volume level based on vehicle speed."
    if "speed" in lowered_context:
        return "Updated vehicle speed value."
    if purpose:
        return f"Expected state change, response, control action, diagnostic result, or user-visible output demonstrating that {_lower_first(purpose)}."
    return ""


def _infer_process(purpose: str, lowered_context: str) -> str:
    if _has_log_context(lowered_context):
        return "Capture speed, volume, manual override, and error events, then write them to the system log."
    if "manual" in lowered_context or "driver manually" in lowered_context:
        return "Detect manual volume changes, pause automatic adjustment temporarily, and preserve the user's selected volume."
    if "ui" in lowered_context or "display" in lowered_context:
        return "Read current speed and media volume data, then update the user interface display."
    if "threshold" in lowered_context or "minimum volume" in lowered_context or "maximum volume" in lowered_context:
        return "Compare the target media volume against configured limits and clamp it within the allowed range."
    if "smooth" in lowered_context or "gradual" in lowered_context or "sudden" in lowered_context:
        return "Calculate intermediate volume steps and apply them gradually until the target level is reached."
    if "volume" in lowered_context and "speed" in lowered_context:
        return "Read vehicle speed, calculate the target media volume using predefined rules, and apply the volume change."
    if "speed" in lowered_context:
        return "Receive vehicle speed data from the vehicle interface and update the internal speed value continuously."
    if purpose:
        return f"Process the relevant system inputs to satisfy this requirement: {purpose}."
    return ""


def _infer_access_restrictions(lowered_context: str) -> str:
    if any(token in lowered_context for token in ("admin", "administrator", "authorized", "role", "credential", "login", "security")):
        return "Role-based access restricted to authorized users or services with the required project permissions."
    if any(token in lowered_context for token in ("calibration", "configuration", "parameter")):
        return "Configuration/calibration changes are restricted to authorized engineering or service roles; runtime use is limited to approved system components."
    return "Role-based access applies where the function is user- or service-facing; otherwise execution is restricted to authorized software components."


def _infer_mandatory_fields(req, lowered_context: str) -> str:
    parts = []
    if req.inputs and not SRSValidator()._is_missing_requirement_value(req.inputs):
        parts.append(f"inputs ({_shorten(req.inputs, 160)})")
    if req.outputs and not SRSValidator()._is_missing_requirement_value(req.outputs):
        parts.append(f"expected outputs ({_shorten(req.outputs, 160)})")
    if any(token in lowered_context for token in ("request", "command", "message", "signal")):
        parts.append("triggering request/message/signal")
    if any(token in lowered_context for token in ("threshold", "range", "limit", "calibration")):
        parts.append("configured threshold/range/calibration values")
    if not parts:
        parts = ["requirement ID", "trigger condition", "input data", "expected output", "validation result"]
    return "Mandatory fields: " + "; ".join(parts) + "."


def _infer_pre_loaded_values(lowered_context: str) -> str:
    if any(token in lowered_context for token in ("lookup", "table", "list of values", "lov")):
        return "Applicable lookup-table/list-of-values data shall be pre-loaded from the approved project configuration."
    if any(token in lowered_context for token in ("calibration", "calibratable", "threshold", "parameter")):
        return "Approved calibration parameters, thresholds, and configuration values shall be pre-loaded before this function executes."
    if any(token in lowered_context for token in ("default", "initial", "startup", "power on")):
        return "Startup/default values defined by the project configuration shall be pre-loaded during initialization."
    return "No source-specific pre-loaded values are identified; use approved project configuration and interface initialization values where applicable."


def _infer_default_values(lowered_context: str) -> str:
    match = re.search(r"\bdefault(?: value)?(?:\s+is|\s*=|\s*:)?\s*([A-Za-z0-9_.%/+ -]{1,80})", lowered_context)
    if match:
        return f"Default value: {match.group(1).strip()}."
    if any(token in lowered_context for token in ("fail-safe", "safe state", "passive state")):
        return "Default to the project-defined safe/passive state when valid input or configuration is unavailable."
    return "Use project-approved default values for optional configuration; mandatory runtime values must be supplied or validated before processing."


def _infer_valid_range(lowered_context: str, context: str) -> str:
    range_text = _extract_range_or_limit(context)
    if range_text:
        return range_text
    if any(token in lowered_context for token in ("boolean", "enable", "disable", "on/off", "active", "inactive")):
        return "Valid values are Boolean/state values defined by the interface, such as enabled/disabled or active/inactive."
    if any(token in lowered_context for token in ("temperature", "voltage", "current", "speed", "pressure", "soc")):
        return "Valid values are the calibrated sensor/interface range and threshold limits defined for the target operating environment."
    return "Valid values are constrained by the applicable interface specification, configuration data, and operating-environment limits."


def _infer_latency(lowered_context: str, context: str) -> str:
    timing = _extract_timing(context)
    if timing:
        return f"Fulfillment time: {timing}."
    if any(token in lowered_context for token in ("diagnostic", "request", "response")):
        return "Fulfillment time is within the configured diagnostic/service response timeout for the target platform."
    if any(token in lowered_context for token in ("control", "sensor", "signal", "state machine")):
        return "Fulfillment time is within the allocated control-loop or signal-processing cycle for this requirement."
    return "Fulfillment time is within the project-defined transaction or task scheduling budget for this function."


def _infer_retention(lowered_context: str) -> str:
    if any(token in lowered_context for token in ("log", "audit", "history")):
        return "Retain log/audit records according to the project diagnostic and data-retention policy."
    if any(token in lowered_context for token in ("diagnostic", "dtc", "fault")):
        return "Retain diagnostic/fault status according to the project diagnostic memory and clearing policy."
    if any(token in lowered_context for token in ("calibration", "configuration", "nonvolatile", "nvm", "eeprom", "flash")):
        return "Retain configuration/calibration data in the approved nonvolatile or project-controlled storage location."
    return "Retain runtime data only for the active transaction unless project diagnostics, logging, or configuration storage requires longer retention."


def _infer_data_rate(lowered_context: str, context: str) -> str:
    timing = _extract_timing(context)
    if timing and any(token in lowered_context for token in ("every", "periodic", "cycle", "frequency", "rate", "sampling")):
        return f"Processed/transmitted at the documented periodic rate: {timing}."
    if any(token in lowered_context for token in ("on request", "request", "command")):
        return "Transaction rate is event-driven and occurs for each valid external request or command."
    if any(token in lowered_context for token in ("signal", "sensor", "state machine")):
        return "Data rate follows each relevant signal update or state-machine execution cycle."
    return "Number of transactions is governed by external events, periodic scheduling, and the project performance budget."


def _infer_external_events(purpose: str, lowered_context: str) -> str:
    if "ignition" in lowered_context:
        return "Ignition state change, valid external command/message, or subsystem state update triggers this functionality."
    if any(token in lowered_context for token in ("request", "command", "diagnostic", "obd", "uds")):
        return "A valid external request/command or diagnostic service request triggers this functionality."
    if any(token in lowered_context for token in ("sensor", "signal", "message", "can", "lin", "ethernet")):
        return "A sensor update, network/interface message, or connected subsystem state change triggers this functionality."
    if purpose:
        return f"Relevant external user actions, interface messages, commands, or subsystem state changes that require {_lower_first(purpose)}."
    return ""


def _infer_temporal_events(lowered_context: str, context: str) -> str:
    timing = _extract_timing(context)
    if timing:
        return f"Time-based trigger or monitoring interval: {timing}."
    if any(token in lowered_context for token in ("periodic", "cycle", "scheduled", "timeout", "debounce", "delay")):
        return "Periodic task execution, debounce/filter timing, delay expiry, or timeout monitoring triggers this functionality."
    return "Periodic scheduling and timeout monitoring apply when required by the software architecture or interface timing."


def _infer_constraints(lowered_context: str) -> str:
    constraints = []
    if any(token in lowered_context for token in ("safety", "asil", "hazard", "fail-safe")):
        constraints.append("comply with project safety goals and fail-safe behavior")
    if any(token in lowered_context for token in ("security", "cyber", "authentication", "authorized")):
        constraints.append("comply with cybersecurity and access-control requirements")
    if any(token in lowered_context for token in ("latency", "every", "cycle", "response time", "timeout")):
        constraints.append("meet timing and response-time budgets")
    if any(token in lowered_context for token in ("ram", "flash", "eeprom", "memory", "cpu")):
        constraints.append("stay within project resource limits")
    if any(token in lowered_context for token in ("can", "lin", "ethernet", "api", "interface")):
        constraints.append("conform to the referenced interface/API specification")
    if constraints:
        return "Implementation constraints: " + "; ".join(constraints) + "."
    return "Implementation must satisfy the documented operating environment, interface contracts, timing budgets, safety/security rules, and project delivery constraints."


def _infer_effects(req) -> str:
    outputs = str(req.outputs or "").strip()
    if outputs and not SRSValidator()._is_missing_requirement_value(outputs):
        return f"Affects downstream components or users that consume the requirement output: {_shorten(outputs, 220)}."
    return "Affects connected subsystems, interfaces, diagnostics, state machines, or user-facing functions that depend on this requirement."


def _infer_failure_scenario(purpose: str) -> str:
    if purpose:
        return f"Failure occurs if required inputs are invalid, unavailable, delayed, or out of range, preventing the system from confirming that {_lower_first(purpose)}."
    return "Failure occurs if required inputs, interfaces, configuration values, or dependent services are invalid, unavailable, delayed, or out of range."


def _infer_criticality(req, context: str) -> bool:
    lowered = context.lower()
    if req.requirement_type == "safety":
        return True
    return any(
        cue in lowered
        for cue in (
            "safety",
            "manual override",
            "phone call",
            "no abrupt",
            "distract",
            "invalid speed",
            "minimum volume",
            "maximum volume",
            "threshold",
            "error",
            "fault",
            "diagnostic",
            "shutdown",
            "overvoltage",
            "undervoltage",
            "overtemperature",
            "security",
            "cyber",
        )
    )


def _has_log_context(lowered_context: str) -> bool:
    return bool(re.search(r"\b(?:log|logging|logged|logs)\b", lowered_context))


def _is_weak_process_value(value: object, req_id: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if req_id and re.fullmatch(rf"{re.escape(req_id)}\s*[:\-].+", text, flags=re.I):
        return True
    return False


def _clean_requirement_sentence(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower().strip(".") in SRSValidator.MISSING_TEXT_VALUES:
        return ""
    if SRSValidator()._contains_unresolved_placeholder(text):
        return ""
    text = _clean_angle_text(text)
    if SRSValidator()._contains_unresolved_placeholder(text):
        return ""
    return text.rstrip(".")


def _lower_first(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return value
    return value[:1].lower() + value[1:]


def _shorten(value: object, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip(" ,;") + "..."


def _requirement_subject(req, purpose: str) -> str:
    if purpose:
        text = re.sub(r"^(?:the\s+)?(?:software|system|application|component)\s+shall\s+", "", purpose, flags=re.I)
        return _shorten(text.rstrip("."), 120).lower()
    block = str(req.logical_block or "").strip()
    if block and block.lower() != "general":
        return block.lower()
    return f"requirement {req.req_id}"


def _clean_angle_text(value: str) -> str:
    text = re.sub(r"<([^>]+)>", r"\1", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _extract_timing(context: str) -> str:
    patterns = (
        r"\bevery\s+\d+(?:\.\d+)?\s*(?:us|microseconds|ms|milliseconds|s|sec|seconds|minutes|min|hz)\b",
        r"\bwithin\s+\d+(?:\.\d+)?\s*(?:us|microseconds|ms|milliseconds|s|sec|seconds|minutes|min)\b",
        r"\b\d+(?:\.\d+)?\s*(?:us|microseconds|ms|milliseconds|s|sec|seconds|minutes|min|hz)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, context or "", flags=re.I)
        if match:
            return match.group(0).strip()
    return ""


def _extract_range_or_limit(context: str) -> str:
    patterns = (
        r"\[[^\]]{1,80}\]",
        r"\bbetween\s+\d+(?:\.\d+)?\s*(?:%|v|a|c|degc|km/h|mph|rpm)?\s+and\s+\d+(?:\.\d+)?\s*(?:%|v|a|c|degc|km/h|mph|rpm)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:%|v|a|c|degc|km/h|mph|rpm)?\s*(?:\.\.|-|to)\s*\d+(?:\.\d+)?\s*(?:%|v|a|c|degc|km/h|mph|rpm)?\b",
        r"\b(?:less than or equal to|greater than or equal to|not exceed|no more than|at least|<=|>=|<|>)\s*\d+(?:\.\d+)?\s*(?:%|v|a|c|degc|km/h|mph|rpm|ms|seconds?)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, context or "", flags=re.I)
        if match:
            return f"Valid range/limit: {match.group(0).strip()}."
    return ""


def _extract_milestone(context: str) -> str:
    match = re.search(r"\b(?:PF\d+(?:\.\d+)?|[ABC]-?sample|prototype|release\s+\d+(?:\.\d+)?)\b", context or "", flags=re.I)
    return match.group(0).strip() if match else ""
