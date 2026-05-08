from __future__ import annotations

from typing import List

from srs_generator.models import SRSProject, ValidationFinding


class SRSValidator:
    REQUIRED_REQUIREMENT_FIELDS = ("purpose", "inputs", "outputs", "process", "validation", "acceptance_criteria")
    REQUIRED_PROJECT_FIELDS = (
        "purpose",
        "intended_audience",
        "system_overview",
        "scope",
        "operating_environment",
        "acceptance_criteria",
    )
    DEFAULT_INTENDED_AUDIENCE = "Project stakeholders, developers, reviewers, testers, and quality teams."

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
                if not value or value == "Not specified.":
                    findings.append(
                        ValidationFinding(
                            severity="warning",
                            code="MISSING_REQUIREMENT_FIELD",
                            message=f"{field.replace('_', ' ').title()} is missing.",
                            req_id=req.req_id,
                            field=field,
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
        if not text or text == "Not specified.":
            return True
        if "<" in text and ">" in text:
            return True
        if field == "intended_audience" and text == self.DEFAULT_INTENDED_AUDIENCE:
            return True
        return False
