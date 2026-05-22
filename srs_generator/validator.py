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
        return bool(self.UNRESOLVED_PATTERN.search(str(value or "")))

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
    """Populate missing SRS fields with a temporary value instead of asking questions.

    The previous clarification-question flow is intentionally disabled for the
    current implementation. This helper keeps the missing-field detection
    internal, writes a temporary answer into text fields, and removes the
    related findings so callers do not surface follow-up questions.
    """
    validator = SRSValidator()
    findings = list(project.validation_findings or validator.validate(project))
    fallback_fields: List[dict[str, str]] = []

    for field in validator.REQUIRED_PROJECT_FIELDS:
        if hasattr(project, field) and validator._is_missing_project_value(field, getattr(project, field, None)):
            setattr(project, field, fallback_text)
            fallback_fields.append({"target": "project", "field": field, "reason": "Missing mandatory project field"})

    for req in project.requirements:
        for field in (*validator.REQUIRED_REQUIREMENT_FIELDS, *validator.OPTIONAL_REQUIREMENT_FIELDS):
            if not hasattr(req, field):
                continue
            value = getattr(req, field)
            if isinstance(value, bool):
                continue
            if field in {"critical", "feasible"} and value is None:
                continue
            if validator._is_missing_requirement_value(value) or validator._contains_unresolved_placeholder(value):
                setattr(req, field, fallback_text)
                req.field_evidence[field] = FieldEvidence(
                    value=fallback_text,
                    confidence=0.0,
                    source_text="",
                    extraction_method="fallback",
                    fallback_reason="Missing or unresolved in source document",
                )
                fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": field, "reason": "Missing or unresolved field"})

        for metadata_field in ("customer_req_id", "milestone", "priority"):
            value = getattr(req.metadata, metadata_field, None)
            if validator._is_missing_text_value(value):
                setattr(req.metadata, metadata_field, fallback_text)
                req.field_evidence[metadata_field] = FieldEvidence(
                    value=fallback_text,
                    confidence=0.0,
                    source_text="",
                    extraction_method="fallback",
                    fallback_reason="Missing metadata field",
                )
                fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": metadata_field, "reason": "Missing metadata field"})

        if req.requirement_type == "safety" and validator._is_missing_text_value(req.metadata.asil):
            req.metadata.asil = fallback_text
            req.field_evidence["asil"] = FieldEvidence(
                value=fallback_text,
                confidence=0.0,
                source_text="",
                extraction_method="fallback",
                fallback_reason="Missing ASIL for safety requirement",
            )
            fallback_fields.append({"target": "requirement", "req_id": req.req_id, "field": "asil", "reason": "Missing ASIL"})

    project.extraction_passes["fallback_fields"] = fallback_fields
    project.validation_findings = [finding for finding in findings if finding.code not in MISSING_FIELD_CODES]
    return project
