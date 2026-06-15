from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DocumentBlock(BaseModel):
    kind: str
    text: str = ""
    heading_level: Optional[int] = None
    style: Optional[str] = None
    rows: List[List[str]] = Field(default_factory=list)
    page: Optional[int] = None
    index: int = 0


class ParsedDocument(BaseModel):
    source_name: str
    source_type: str
    text: str
    blocks: List[DocumentBlock] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequirementMetadata(BaseModel):
    customer_req_id: Optional[str] = None
    milestone: Optional[str] = None
    priority: Optional[str] = None
    asil: Optional[str] = None
    cal: Optional[str] = None
    can_ids: List[str] = Field(default_factory=list)
    sds_references: List[str] = Field(default_factory=list)
    source_section: Optional[str] = None
    source_block_index: Optional[int] = None
    extraction_method: str = "hybrid"
    evidence: Optional[str] = None
    confidence: float = 0.0


class FieldEvidence(BaseModel):
    value: str = ""
    confidence: float = 0.0
    source_text: str = ""
    extraction_method: str = "heuristic"
    fallback_reason: Optional[str] = None


class RevisionHistoryEntry(BaseModel):
    version: str = ""
    author: str = ""
    author_date: str = ""
    reviewer: str = ""
    reviewer_date: str = ""
    approver: str = ""
    approver_date: str = ""
    change_description: str = ""
    confidence: float = 0.0
    source_text: str = ""


class DefinitionEntry(BaseModel):
    term: str = ""
    description: str = ""
    confidence: float = 0.0
    source_text: str = ""


class ReferenceEntry(BaseModel):
    number: str = ""
    document: str = ""
    version: str = ""
    remarks: str = ""
    confidence: float = 0.0
    source_text: str = ""


class StructuredRequirement(BaseModel):
    logical_block: str = "General"
    req_id: str
    requirement_type: str = "functional"
    purpose: str = "Not specified."
    inputs: str = "Not specified."
    outputs: str = "Not specified."
    process: str = "Not specified."
    validation: str = "Not specified."
    acceptance_criteria: str = "Not specified."
    derived_requirement: str = "No derived requirement identified."
    access_restrictions: str = "No specific access restrictions identified."
    mandatory_fields: str = "Not specified."
    pre_loaded_values: str = "Not specified."
    default_values: str = "Not specified."
    valid_range_of_values: str = "Not specified."
    data_latency_period: str = "Not specified."
    data_retention_period: str = "Not specified."
    data_rate: str = "Not specified."
    external_events: str = "Not specified."
    temporal_events: str = "Not specified."
    constraints: str = "Not specified."
    effects_on_other_systems: str = "Not specified."
    assumptions: str = "Not specified."
    failure_scenario: str = "Not specified."
    action_on_failure: str = "Not specified."
    testability: str = "Not specified."
    status: str = "Proposed"
    critical: Optional[bool] = None
    feasible: Optional[bool] = None
    comments: str = ""
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: RequirementMetadata = Field(default_factory=RequirementMetadata)
    field_evidence: Dict[str, FieldEvidence] = Field(default_factory=dict)

    @validator("req_id")
    def req_id_must_not_be_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("req_id is required")
        return value

    @validator("logical_block", "requirement_type", pre=True, always=True)
    def normalize_short_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "General"


class ValidationFinding(BaseModel):
    severity: str
    code: str
    message: str
    req_id: Optional[str] = None
    field: Optional[str] = None


class SRSProject(BaseModel):
    project_name: str = "Unnamed Project"
    version: str = "1.0"
    document_title: str = "Software Requirement Specification"
    system_overview: str = "Not specified."
    scope: str = "Not specified."
    purpose: str = "This document captures software requirements for the project."
    intended_audience: str = "Project stakeholders, developers, reviewers, testers, and quality teams."
    diagnostics: str = "Not specified."
    safety_requirements: List[StructuredRequirement] = Field(default_factory=list)
    cybersecurity_requirements: List[StructuredRequirement] = Field(default_factory=list)
    interfaces: Dict[str, str] = Field(default_factory=dict)
    acceptance_criteria: str = "Not specified."
    operating_environment: str = "Not specified."
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    requirements: List[StructuredRequirement] = Field(default_factory=list)
    revision_history: List[RevisionHistoryEntry] = Field(default_factory=list)
    definitions: List[DefinitionEntry] = Field(default_factory=list)
    references: List[ReferenceEntry] = Field(default_factory=list)
    extraction_passes: Dict[str, Any] = Field(default_factory=dict)
    validation_findings: List[ValidationFinding] = Field(default_factory=list)
    source_name: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")  # type: ignore[attr-defined]
        return self.dict()
