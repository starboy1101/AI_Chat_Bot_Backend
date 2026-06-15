from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from srs_generator.models import (
    DefinitionEntry,
    FieldEvidence,
    ParsedDocument,
    ReferenceEntry,
    RequirementMetadata,
    RevisionHistoryEntry,
    SRSProject,
    StructuredRequirement,
)
from srs_generator.parser import DocumentParser
from srs_generator.utils import compact_inline, normalize_space, truthy_text
from srs_generator.validator import SRSValidator


REQ_ID_PATTERN = re.compile(
    r"\b(?:SWRS|SWR|REQ|RQ|FR|NFR|BR|UR|SAFREQ|SW_SAF_REQ|CS_REQ|CS-REQ)[A-Z0-9_-]*[_-]?\d+[A-Z0-9_-]*\b",
    re.I,
)
SDS_PATTERN = re.compile(r"\bSDS[-_A-Z0-9]*-\d+\b", re.I)
MILESTONE_PATTERN = re.compile(
    r"\b(?:belongs\s+to\s+)?(?:milestone|phase|release)\s*(?:is|=|:|-|named|as)?\s*"
    r"([A-Z][A-Z0-9_.-]*(?:/[A-Z0-9_.-]+)?)\b",
    re.I,
)
PRIORITY_PATTERN = re.compile(r"\b(?:priority|classified\s+as)\s*(?:is|=|:|-)?\s*(urgent|high|medium|low)\b", re.I)
CAN_ID_PATTERN = re.compile(r"\b(?:0x[0-9A-Fa-f]{2,8}|CAN\s*ID\s*[:#-]?\s*[0-9A-Fa-fx]+)\b")
ASIL_PATTERN = re.compile(r"\bASIL\s*(?:QM|[ABCD])\b", re.I)
CAL_PATTERN = re.compile(r"\bCAL\s*[1-4]\b", re.I)
VERSION_PATTERN = re.compile(r"\b(?:version|baseline|draft)\s*[: ]\s*([0-9]+(?:\.[0-9]+)*)\b", re.I)
HEADING_NUMBER_PREFIX = re.compile(r"^\d+(?:\.\d+)*\.?\s*")
SIMPLE_REQ_HEADING_PATTERN = re.compile(
    r"^\s*((?:FR|NFR|BR|UR|REQ)[-_]?\d+[A-Z0-9_-]*)\s+(.+?)\s*$",
    re.I,
)
FIELD_LABEL_PATTERN = re.compile(
    r"(?im)^\s*(req(?:uirement)? id|customer req(?:uirement)? id|milestone|release|phase|priority|description|purpose|input(?:\(s\))?|inputs|output(?:\(s\))?|outputs|process(?: flow)?|workflow|sequence|validation(?: rules)?|verification criteria|acceptance criteria|constraints?|assumptions?|pre[- ]loaded values|default values|valid range(?: of values)?|data latency period|data retention period|data rate|external events|temporal events|effects on other systems(?:/sub system)?|testability(?: with respect to test environment)?)\s*:?\s*$"
)
INLINE_FIELD_LABEL_PATTERN = re.compile(
    r"(?im)^\s*(req(?:uirement)? id|customer req(?:uirement)? id|milestone|release|phase|priority|description|purpose|input(?:\(s\))?|inputs|output(?:\(s\))?|outputs|process(?: flow)?|workflow|sequence|validation(?: rules)?|verification criteria|acceptance criteria|constraints?|assumptions?|pre[- ]loaded values|default values|valid range(?: of values)?|data latency period|data retention period|data rate|external events|temporal events|effects on other systems(?:/sub system)?|testability(?: with respect to test environment)?)\s*[:\-]\s*(.+?)\s*$"
)
BULLET_PREFIX_PATTERN = re.compile(r"^\s*(?:[•*+\-–—]|\d+[.)]|[a-zA-Z][.)])\s+")
NUMBERED_STEP_PATTERN = re.compile(r"^\s*(?:\d+[.)]|\([a-zA-Z0-9]+\)|[a-zA-Z][.)])\s+")
REQUIREMENT_MODAL_PATTERN = re.compile(
    r"\b(?:shall|should|must|will|can|allow|allows|enable|enables|support|supports|provide|provides|track|tracks|manage|manages|display|displays|store|stores|validate|validates|respond|responds)\b",
    re.I,
)
REQUIREMENT_SECTION_PATTERN = re.compile(
    r"\b(?:requirements?|functional requirements?|non[- ]functional requirements?|features?|scope|system flow|use cases?|capabilities)\b",
    re.I,
)
NON_FUNCTIONAL_PATTERN = re.compile(
    r"\b(?:non[- ]functional|performance|security|usability|availability|reliability|scalability|maintainability|response time|startup time|memory utilization|cpu utilization)\b",
    re.I,
)
HEADING_LIKE_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S+|^[A-Z][A-Za-z0-9 /&()_-]{2,80}:?\s*$")
FIELD_BOUNDARY_CUES = (
    "derived requirement",
    "requirement priority",
    "access restriction",
    "input condition",
    "inputs required",
    "outputs expected",
    "outputs include",
    "output behavior",
    "mandatory field",
    "pre-loaded",
    "pre loaded",
    "preloaded",
    "default value",
    "valid range",
    "data latency",
    "data retention",
    "data rate",
    "external event",
    "temporal event",
    "validation",
    "verification",
    "acceptance",
    "constraint",
    "effect on other",
    "effects on other",
    "assumption",
    "failure scenario",
    "action if failure",
    "testability",
    "feasible",
    "comment",
    "requirement status",
)
METADATA_SENTENCE_CUES = (
    "corresponds to customer requirement",
    "belongs to milestone",
    "derived requirement",
    "requirement priority",
    "categorized as",
    "classified as",
    "considered critical",
    "requirement is considered",
    "access restriction",
    "no specific access",
    "no derived requirement",
    "not applicable",
    "is applicable",
)
VALIDATION_SENTENCE_CUES = (
    "validation criteria",
    "validation activities",
    "verification criteria",
    "verify ",
    "validate ",
    "tested using",
    "testing",
    "simulation",
    "fault injection",
    "canalyzer",
    "canoe",
    "canape",
    "hil ",
    " hil",
    "sil ",
    " sil",
)
ACCEPTANCE_SENTENCE_CUES = ("acceptance criteria", "accepted when", "pass criteria", "shall be accepted")
OUTPUT_SENTENCE_CUES = (
    "outputs expected",
    "output behavior",
    "outputs include",
    "shall broadcast",
    "shall transmit every",
    "pdu shall transmit",
    "transmission of",
)
NON_FUNCTIONAL_PURPOSE_CUES = (
    "startup time",
    "response time",
    "memory utilization",
    "cpu utilization",
    "performance",
    "availability",
    "reliability",
    "scalability",
    "maintainability",
)

SECTION_KEYWORDS = {
    "communication": {"communication", "can", "lin", "ethernet", "dbc", "arxml", "message", "signal"},
    "diagnostics": {"diagnostic", "dtc", "obd", "fault", "uds", "dem", "debounce"},
    "safety": {"safety", "asil", "safe state", "ftti", "fmea", "safreq"},
    "cybersecurity": {"cyber", "security", "cal", "threat", "vulnerability", "audit"},
    "interfaces": {"interface", "hardware", "software", "api", "driver", "input", "output"},
    "state-machine": {"state", "mode", "transition", "state-machine", "state machine"},
    "operating_environment": {"environment", "platform", "operating", "compiler", "toolchain"},
    "acceptance": {"acceptance", "test scenario", "verification", "criteria"},
}

LABEL_ALIASES = {
    "req_id": {"req id", "requirement id", "id"},
    "customer_req_id": {"customer req id", "customer requirement id", "source requirement id"},
    "milestone": {"milestone", "release", "phase"},
    "purpose": {"purpose", "purpose1", "description", "requirement description", "feature", "functionality", "detailed requirements specification", "detailed requirement specification"},
    "derived_requirement": {"derived requirement", "derived from"},
    "priority": {"requirement priority", "priority"},
    "access_restrictions": {"access restrictions", "access restriction"},
    "inputs": {"inputs", "input(s)", "input"},
    "outputs": {"outputs", "output(s)", "output"},
    "process": {"process", "workflow", "sequence"},
    "mandatory_fields": {"mandatory fields", "mandatory field"},
    "pre_loaded_values": {"pre-loaded values", "pre loaded values", "preloaded values"},
    "default_values": {"default values", "default value"},
    "valid_range_of_values": {"valid range of values", "valid range of value", "valid range"},
    "data_latency_period": {"data latency period", "data latency"},
    "data_retention_period": {"data retention period", "data retention"},
    "data_rate": {"data rate daily number of transaction", "data rate daily number of transactions", "data rate number of transaction", "data rate number of transactions", "data rate"},
    "external_events": {"external events", "external event"},
    "temporal_events": {"temporal events", "temporal event"},
    "validation": {"validation rules verification criteria", "validation rules", "verification criteria", "verification method"},
    "constraints": {"constraints", "constraint"},
    "effects_on_other_systems": {"effects on other systems sub system", "effects on other system", "effect on other system", "effects on other systems"},
    "assumptions": {"assumptions", "assumption"},
    "failure_scenario": {"failure scenario", "fault scenario"},
    "action_on_failure": {"action if failure", "actions if failure", "action on failure"},
    "acceptance_criteria": {"acceptance criteria", "requirement acceptance criteria"},
    "testability": {"testability with respect to test environment", "testability"},
    "status": {"requirement status", "status"},
    "critical": {"critical", "critical (yes no)"},
    "feasible": {"feasible", "feasible within project constraints"},
    "comments": {"comments", "remarks", "rationale"},
    "asil": {"asil level", "asil"},
    "cal": {"cal level applicable", "cal"},
}

FIELD_CUES: Dict[str, Tuple[str, ...]] = {
    "purpose": (
        "this requirement specifies",
        "purpose",
        "shall support",
        "shall provide",
        "shall detect",
        "shall store",
        "shall maintain",
        "shall monitor",
        "shall manage",
        "shall implement",
        "shall acquire",
        "shall disable",
        "shall enable",
        "shall comply",
        "shall not exceed",
    ),
    "inputs": (
        "input condition",
        "inputs required",
        "input(s)",
        "input required",
        "receives",
        "provided by",
        "configuration files",
    ),
    "outputs": (
        "outputs expected",
        "output behavior",
        "output(s)",
        "shall broadcast",
        "shall transmit",
        "transmission of",
        "state machine status",
    ),
    "process": (
        "process flow",
        "process specifies",
        "process requires",
        "the process requires",
        "workflow",
        "sequence",
        "steps",
        "transition",
    ),
    "validation": (
        "validation criteria",
        "validation activities",
        "verification criteria",
        "verify ",
        "validate ",
        "tested using",
        "hil ",
        " hil",
        "sil ",
        " sil",
        "canalyzer",
        "canape",
        "fault injection",
    ),
    "acceptance_criteria": (
        "acceptance criteria",
        "accepted when",
        "acceptance",
        "pass criteria",
    ),
    "pre_loaded_values": ("pre-loaded", "preloaded", "boot into", "initial value"),
    "default_values": ("default value", "default values", "default state"),
    "valid_range_of_values": ("valid range", "allowed range", "range for", "minimum", "maximum"),
    "constraints": ("constraint", "shall not", "limited to", "must not"),
    "effects_on_other_systems": ("effects on other", "impact on", "affecting"),
    "assumptions": ("assumption", "assume", "assumed"),
    "failure_scenario": ("failure scenario", "fault scenario", "can go wrong", "failure"),
    "action_on_failure": ("action if failure", "on failure", "fallback", "recovery", "retry"),
    "testability": ("testability", "tested in", "test environment", "test phase"),
    "data_latency_period": ("latency", "response time", "within", "milliseconds", "ms"),
    "data_retention_period": ("retention", "retain", "retained", "stored for", "maintained for", "nonvolatile"),
    "data_rate": ("data rate", "frequency", "frames per second", "transactions"),
    "external_events": ("external event", "triggered by", "trigger", "whenever", "when ", "after ignition", "command"),
    "temporal_events": ("temporal event", "periodic", "scheduled", "time-based", "every ", "cycle time"),
}


class SectionClassifier:
    def classify(self, text: str) -> str:
        lowered = compact_inline(text).lower()
        if not lowered:
            return "general"
        scores: Dict[str, int] = {}
        for section, keywords in SECTION_KEYWORDS.items():
            scores[section] = sum(1 for keyword in keywords if keyword in lowered)
        best_section, best_score = max(scores.items(), key=lambda item: item[1])
        return best_section if best_score else "general"


class RequirementExtractor:
    def __init__(self) -> None:
        self.classifier = SectionClassifier()

    def extract(self, parsed: ParsedDocument) -> SRSProject:
        sections = self._detect_sections(parsed)
        project = SRSProject(
            project_name=self._extract_project_name(parsed),
            version=self._extract_version(parsed.text),
            source_name=parsed.source_name,
            system_overview=self._extract_section_text(
                parsed,
                {"overview", "system overview", "software overview", "overall description", "product perspective"},
            ),
            scope=self._extract_section_text(parsed, {"scope"}),
            purpose=self._extract_section_text(parsed, {"purpose of the document", "purpose", "introduction"}),
            intended_audience=self._extract_section_text(parsed, {"intended audience", "audience"}),
            operating_environment=self._extract_section_text(parsed, {"operating environment"}),
            acceptance_criteria=self._extract_section_text(parsed, {"overall acceptance criteria", "acceptance criteria"}),
            diagnostics=self._extract_section_text(parsed, {"diagnostics", "general diagnostics"}),
            interfaces={
                "user": self._extract_section_text(parsed, {"user interface", "external interfaces"}),
                "hardware": self._extract_section_text(parsed, {"hardware interface", "external interfaces"}),
                "software": self._extract_section_text(parsed, {"software interface", "external interfaces"}),
            },
        )

        project.revision_history = self._extract_revision_history(parsed)
        project.definitions = self._extract_definitions(parsed)
        project.references = self._extract_references(parsed)

        table_requirements = self._extract_table_requirements(parsed)
        text_requirements = self._extract_text_requirements(parsed)
        requirements = self._merge_requirement_sources(table_requirements, text_requirements)

        project.requirements = self._dedupe_requirements(requirements)
        project.safety_requirements = [r for r in project.requirements if r.requirement_type == "safety"]
        project.cybersecurity_requirements = [r for r in project.requirements if r.requirement_type == "cybersecurity"]
        project.assumptions = self._extract_assumptions(parsed)
        project.open_questions = self._extract_open_questions(parsed)
        project.extraction_passes = {
            "pass_1_sections": len(sections),
            "pass_2_requirement_blocks": len(self._detect_requirement_blocks(parsed)),
            "pass_3_table_requirements": len(table_requirements),
            "pass_3_text_requirements": len(text_requirements),
            "revision_history_entries": len(project.revision_history),
            "definitions": len(project.definitions),
            "references": len(project.references),
        }
        project.confidence = self._project_confidence(project)
        project.validation_findings = SRSValidator().validate(project)
        return project

    def _detect_sections(self, parsed: ParsedDocument) -> List[Dict[str, Any]]:
        """Pass 1: preserve document hierarchy instead of flattening pages/paragraphs."""
        sections: List[Dict[str, Any]] = []
        stack: List[Tuple[int, str]] = []
        current: Optional[Dict[str, Any]] = None

        for block in parsed.blocks:
            if block.kind != "heading":
                continue
            heading = self._clean_heading_text(block.text)
            if not heading or self._is_template_heading(heading):
                continue
            level = block.heading_level or 9
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            if current is not None:
                current["end_block_index"] = block.index - 1
            current = {
                "heading": heading,
                "level": level,
                "path": " > ".join(item[1] for item in stack),
                "section_type": self.classifier.classify(" ".join(item[1] for item in stack)),
                "start_block_index": block.index,
                "end_block_index": None,
            }
            sections.append(current)

        if current is not None:
            current["end_block_index"] = parsed.blocks[-1].index if parsed.blocks else None
        return sections

    def _detect_requirement_blocks(self, parsed: ParsedDocument) -> List[Dict[str, Any]]:
        """Pass 2: build semantic requirement chunks with hierarchy and nearby context."""
        blocks: List[Dict[str, Any]] = []
        heading_stack: List[Tuple[int, str]] = []
        current: Optional[Dict[str, Any]] = None

        def close_current() -> None:
            nonlocal current
            if current and normalize_space("\n".join(current["parts"])):
                current["text"] = normalize_space("\n".join(current["parts"]))
                blocks.append(current)
            current = None

        for block in parsed.blocks:
            if block.kind == "heading":
                heading = self._clean_heading_text(block.text)
                level = block.heading_level or 9
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                if heading and not self._is_template_heading(heading):
                    heading_stack.append((level, heading))
                if current and level <= 2 and not REQ_ID_PATTERN.search(block.text or ""):
                    close_current()
                continue

            if block.kind == "table":
                if current and block.text:
                    current["parts"].append(block.text)
                continue

            text = normalize_space(block.text)
            if not text or self._looks_like_template_placeholder(text, {"text": text}):
                continue

            matches = list(REQ_ID_PATTERN.finditer(text))
            if matches:
                for match_index, match in enumerate(matches):
                    if current:
                        close_current()
                    req_id = match.group(0).strip()
                    start = match.start()
                    end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
                    segment = normalize_space(text[start:end])
                    heading_path = " > ".join(item[1] for item in heading_stack) or "General"
                    current = {
                        "req_id": req_id,
                        "parts": [segment],
                        "logical_block": self._requirement_block_from_heading_path(heading_path, segment),
                        "heading_path": heading_path,
                        "section_type": self.classifier.classify(f"{heading_path} {segment}"),
                        "source_block_index": block.index,
                    }
                continue

            if current:
                if self._is_hard_section_break(text):
                    close_current()
                    continue
                current["parts"].append(text)

        close_current()

        if blocks:
            return blocks

        # Fallback for plain text where parser did not preserve block boundaries.
        matches = list(REQ_ID_PATTERN.finditer(parsed.text or ""))
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(parsed.text), match.end() + 3600)
            chunk = normalize_space(parsed.text[start:end])
            if len(chunk) < 30:
                continue
            blocks.append(
                {
                    "req_id": match.group(0).strip(),
                    "text": chunk,
                    "logical_block": self._infer_logical_block(match.group(0), chunk),
                    "heading_path": "General",
                    "section_type": self.classifier.classify(chunk),
                    "source_block_index": None,
                }
            )
        return blocks

    def _requirement_block_from_heading_path(self, heading_path: str, text: str) -> str:
        headings = [item.strip() for item in (heading_path or "").split(">") if item.strip()]
        for heading in reversed(headings):
            cleaned = HEADING_NUMBER_PREFIX.sub("", heading).strip()
            lowered = cleaned.lower()
            if lowered not in {"requirements", "functional requirement", "functional requirements"}:
                return cleaned
        return self._infer_logical_block("", text)

    def _is_hard_section_break(self, text: str) -> bool:
        lowered = HEADING_NUMBER_PREFIX.sub("", text.strip()).lower()
        return lowered in {
            "revision & approval history",
            "definition, acronyms and abbreviations",
            "references",
            "acceptance criteria",
            "assumptions",
            "queries on requirement",
            "risk & opportunity analysis",
            "appendix",
        }

    def _extract_table_requirements(self, parsed: ParsedDocument) -> List[StructuredRequirement]:
        requirements: List[StructuredRequirement] = []
        current_heading = "General"
        current_section_type = "general"

        for block in parsed.blocks:
            if block.kind == "heading":
                heading = normalize_space(block.text)
                if heading and not self._is_template_heading(heading):
                    current_heading = HEADING_NUMBER_PREFIX.sub("", heading).strip() or current_heading
                    current_section_type = self.classifier.classify(current_heading)
                continue
            if block.kind != "table" or not block.rows:
                continue

            segment_requirements: List[StructuredRequirement] = []
            for segment_rows in self._split_requirement_table_segments(block.rows):
                req = self._requirement_from_table(segment_rows, current_heading, current_section_type, block.index)
                if req:
                    segment_requirements.append(req)
            if segment_requirements:
                requirements.extend(segment_requirements)
                continue

            req = self._requirement_from_table(block.rows, current_heading, current_section_type, block.index)
            if req:
                requirements.append(req)
                continue
            requirements.extend(
                self._requirements_from_horizontal_table(
                    block.rows,
                    current_heading,
                    current_section_type,
                    block.index,
                    len(requirements),
                )
            )
        return requirements

    def _requirement_from_table(
        self,
        rows: List[List[str]],
        logical_block: str,
        section_type: str,
        block_index: int,
    ) -> Optional[StructuredRequirement]:
        fields: Dict[str, str] = {}
        for row in rows:
            if len(row) < 2:
                continue
            label = self._normalize_label(row[0])
            key = self._field_key(label)
            if not key:
                continue
            value = self._value_from_labeled_row(row, key)
            if value:
                fields[key] = value

        req_id = fields.get("req_id")
        if not req_id or not self._looks_like_requirement(fields):
            return None
        if self._looks_like_template_placeholder(req_id, fields):
            return None

        evidence = "\n".join(" | ".join(cell for cell in row if cell) for row in rows[:12])
        requirement_type = self._requirement_type(req_id, logical_block, evidence, section_type)
        fields, inferred_fields = self._prepare_requirement_fields(fields, evidence, req_id, requirement_type)
        metadata = self._metadata_from_text(
            text=evidence,
            fields=fields,
            logical_block=logical_block,
            block_index=block_index,
            method="table",
            confidence=0.92,
        )
        field_evidence = self._evidence_from_fields(fields, evidence, "table", 0.92)
        field_evidence.update(self._evidence_from_inferred_fields(inferred_fields, evidence))

        return StructuredRequirement(
            logical_block=self._clean_logical_block(logical_block, requirement_type),
            req_id=req_id.strip(),
            requirement_type=requirement_type,
            purpose=fields.get("purpose") or "Not specified.",
            inputs=fields.get("inputs") or "Not specified.",
            outputs=fields.get("outputs") or "Not specified.",
            process=fields.get("process") or "Not specified.",
            validation=fields.get("validation") or "Not specified.",
            acceptance_criteria=fields.get("acceptance_criteria") or "Not specified.",
            derived_requirement=fields.get("derived_requirement") or "No derived requirement identified.",
            access_restrictions=fields.get("access_restrictions") or "No specific access restrictions identified.",
            mandatory_fields=fields.get("mandatory_fields") or "Not specified.",
            pre_loaded_values=fields.get("pre_loaded_values") or "Not specified.",
            default_values=fields.get("default_values") or "Not specified.",
            valid_range_of_values=fields.get("valid_range_of_values") or "Not specified.",
            data_latency_period=fields.get("data_latency_period") or "Not specified.",
            data_retention_period=fields.get("data_retention_period") or "Not specified.",
            data_rate=fields.get("data_rate") or "Not specified.",
            external_events=fields.get("external_events") or "Not specified.",
            temporal_events=fields.get("temporal_events") or "Not specified.",
            constraints=fields.get("constraints") or "Not specified.",
            effects_on_other_systems=fields.get("effects_on_other_systems") or "Not specified.",
            assumptions=fields.get("assumptions") or "Not specified.",
            failure_scenario=fields.get("failure_scenario") or "Not specified.",
            action_on_failure=fields.get("action_on_failure") or "Not specified.",
            testability=fields.get("testability") or "Not specified.",
            status=fields.get("status") or "Proposed",
            critical=truthy_text(fields.get("critical")),
            feasible=truthy_text(fields.get("feasible")),
            comments=fields.get("comments") or "",
            metadata=metadata,
            field_evidence=field_evidence,
        )

    def _requirements_from_horizontal_table(
        self,
        rows: List[List[str]],
        logical_block: str,
        section_type: str,
        block_index: int,
        existing_count: int,
    ) -> List[StructuredRequirement]:
        if len(rows) < 2:
            return []

        headers = [self._normalize_label(cell) for cell in rows[0]]
        keys = [self._field_key(header) for header in headers]
        if not any(keys):
            return []
        if "req_id" not in keys and not REQUIREMENT_SECTION_PATTERN.search(logical_block or ""):
            return []

        requirements: List[StructuredRequirement] = []
        counters: Dict[str, int] = defaultdict(int)
        if existing_count:
            counters["AUTO-FR"] = existing_count

        for row_index, row in enumerate(rows[1:], start=1):
            fields: Dict[str, str] = {}
            for cell_index, cell in enumerate(row):
                if cell_index >= len(keys):
                    continue
                key = keys[cell_index]
                value = normalize_space(cell)
                if key and value:
                    fields[key] = value

            if not self._looks_like_requirement(fields):
                continue

            row_text = " | ".join(normalize_space(cell) for cell in row if normalize_space(cell))
            req_id = fields.get("req_id")
            requirement_type = self._requirement_type(req_id or "", logical_block, row_text, section_type)
            if not req_id:
                req_id = self._generated_req_id(requirement_type, counters)
            fields, inferred_fields = self._prepare_requirement_fields(fields, row_text, req_id, requirement_type)

            metadata = self._metadata_from_text(
                text=row_text,
                fields=fields,
                logical_block=logical_block,
                block_index=block_index,
                method="horizontal-table",
                confidence=0.82,
            )
            metadata.source_block_index = block_index
            evidence_suffix = f"\nTable row: {row_index}"
            metadata.evidence = normalize_space(f"{metadata.evidence or ''}{evidence_suffix}")
            field_evidence = self._evidence_from_fields(fields, row_text, "horizontal-table", 0.82)
            field_evidence.update(self._evidence_from_inferred_fields(inferred_fields, row_text))

            requirements.append(
                StructuredRequirement(
                    logical_block=self._clean_logical_block(logical_block, requirement_type),
                    req_id=req_id.strip(),
                    requirement_type=requirement_type,
                    purpose=fields.get("purpose") or "Not specified.",
                    inputs=fields.get("inputs") or "Not specified.",
                    outputs=fields.get("outputs") or "Not specified.",
                    process=fields.get("process") or "Not specified.",
                    validation=fields.get("validation") or "Not specified.",
                    acceptance_criteria=fields.get("acceptance_criteria") or "Not specified.",
                    derived_requirement=fields.get("derived_requirement") or "No derived requirement identified.",
                    access_restrictions=fields.get("access_restrictions") or "No specific access restrictions identified.",
                    mandatory_fields=fields.get("mandatory_fields") or "Not specified.",
                    pre_loaded_values=fields.get("pre_loaded_values") or "Not specified.",
                    default_values=fields.get("default_values") or "Not specified.",
                    valid_range_of_values=fields.get("valid_range_of_values") or "Not specified.",
                    data_latency_period=fields.get("data_latency_period") or "Not specified.",
                    data_retention_period=fields.get("data_retention_period") or "Not specified.",
                    data_rate=fields.get("data_rate") or "Not specified.",
                    external_events=fields.get("external_events") or "Not specified.",
                    temporal_events=fields.get("temporal_events") or "Not specified.",
                    constraints=fields.get("constraints") or "Not specified.",
                    effects_on_other_systems=fields.get("effects_on_other_systems") or "Not specified.",
                    assumptions=fields.get("assumptions") or "Not specified.",
                    testability=fields.get("testability") or "Not specified.",
                    status=fields.get("status") or "Proposed",
                    critical=truthy_text(fields.get("critical")),
                    feasible=truthy_text(fields.get("feasible")),
                    comments=fields.get("comments") or "",
                    metadata=metadata,
                    field_evidence=field_evidence,
                )
            )

        return requirements

    def _split_requirement_table_segments(self, rows: List[List[str]]) -> List[List[List[str]]]:
        starts = [
            index
            for index in range(len(rows))
            if self._is_requirement_segment_start(rows, index)
        ]
        if not starts:
            return []
        segments: List[List[List[str]]] = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(rows)
            segment = rows[start:end]
            if len(segment) >= 2:
                segments.append(segment)
        return segments

    def _is_requirement_segment_start(self, rows: List[List[str]], index: int) -> bool:
        row = rows[index]
        if not row or self._normalize_label(row[0]) != "term":
            return False
        for lookahead in rows[index + 1 : index + 4]:
            labels = [self._normalize_label(cell) for cell in lookahead[:4]]
            if any(label in {"req id", "requirement id"} for label in labels):
                return True
        return False

    def _extract_text_requirements(self, parsed: ParsedDocument) -> List[StructuredRequirement]:
        requirements: List[StructuredRequirement] = []
        requirement_blocks = self._detect_requirement_blocks(parsed)
        if not requirement_blocks:
            return self._extract_implicit_requirements(parsed)

        for block in requirement_blocks:
            chunk = normalize_space(block.get("text") or "")
            if len(chunk) < 30:
                continue
            req_id = str(block.get("req_id") or "").strip()
            if self._looks_like_template_placeholder(req_id, {"text": chunk}):
                continue
            logical_block = str(block.get("logical_block") or self._infer_logical_block(req_id, chunk))
            section_type = str(block.get("section_type") or self.classifier.classify(chunk))
            requirement_type = self._requirement_type(req_id, logical_block, chunk, section_type)
            fields, field_evidence = self._fields_from_semantic_block(chunk)
            fields, inferred_fields = self._prepare_requirement_fields(fields, chunk, req_id, requirement_type)
            field_evidence.update(self._evidence_from_inferred_fields(inferred_fields, chunk))
            title = self._title_from_text_requirement(req_id, chunk)
            if title:
                logical_block = title
            metadata = self._metadata_from_text(
                text=chunk,
                fields=fields,
                logical_block=logical_block,
                block_index=block.get("source_block_index"),
                method="semantic-text",
                confidence=0.86 if fields else 0.68,
            )
            if not metadata.customer_req_id and metadata.sds_references:
                metadata.customer_req_id = metadata.sds_references[0]
            requirements.append(
                StructuredRequirement(
                    logical_block=logical_block,
                    req_id=req_id,
                    requirement_type=requirement_type,
                    purpose=fields.get("purpose") or self._best_purpose_text("", chunk, req_id, requirement_type) or chunk[:500],
                    inputs=fields.get("inputs") or "Not specified.",
                    outputs=fields.get("outputs") or "Not specified.",
                    process=fields.get("process") or "Not specified.",
                    validation=fields.get("validation") or "Not specified.",
                    acceptance_criteria=fields.get("acceptance_criteria") or "Not specified.",
                    pre_loaded_values=fields.get("pre_loaded_values") or "Not specified.",
                    default_values=fields.get("default_values") or "Not specified.",
                    valid_range_of_values=fields.get("valid_range_of_values") or "Not specified.",
                    data_latency_period=fields.get("data_latency_period") or "Not specified.",
                    data_retention_period=fields.get("data_retention_period") or "Not specified.",
                    data_rate=fields.get("data_rate") or "Not specified.",
                    external_events=fields.get("external_events") or "Not specified.",
                    temporal_events=fields.get("temporal_events") or "Not specified.",
                    constraints=fields.get("constraints") or "Not specified.",
                    effects_on_other_systems=fields.get("effects_on_other_systems") or "Not specified.",
                    assumptions=fields.get("assumptions") or "Not specified.",
                    testability=fields.get("testability") or "Not specified.",
                    critical=truthy_text("critical" if "critical" in chunk.lower() else ""),
                    feasible=truthy_text("yes" if "feasible" in chunk.lower() else ""),
                    metadata=metadata,
                    field_evidence=field_evidence,
                )
            )
        return requirements

    def _extract_implicit_requirements(self, parsed: ParsedDocument) -> List[StructuredRequirement]:
        labeled_requirements = self._extract_labeled_text_blocks(parsed)
        if labeled_requirements:
            return labeled_requirements

        requirements: List[StructuredRequirement] = []
        current_section = "General"
        counters: Dict[str, int] = defaultdict(int)
        seen: set[str] = set()

        for line in self._iter_text_lines(parsed):
            clean_line = normalize_space(line)
            if not clean_line:
                continue

            if self._is_probable_heading(clean_line):
                current_section = self._clean_heading_text(clean_line)
                continue

            was_bullet = bool(BULLET_PREFIX_PATTERN.match(clean_line))
            candidate = self._clean_requirement_candidate(clean_line)
            if not self._is_implicit_requirement_candidate(candidate, current_section, was_bullet):
                continue

            purpose = self._requirement_sentence(candidate)
            key = compact_inline(purpose).lower()
            if not purpose or key in seen:
                continue
            seen.add(key)

            requirement_type = self._implicit_requirement_type(current_section, purpose)
            req_id = self._generated_req_id(requirement_type, counters)
            logical_block = self._implicit_logical_block(current_section, purpose, requirement_type)
            section_type = self.classifier.classify(f"{current_section} {purpose}")
            metadata = self._metadata_from_text(
                text=purpose,
                fields={},
                logical_block=logical_block,
                block_index=None,
                method="implicit-text",
                confidence=0.56 if was_bullet else 0.5,
            )

            requirements.append(
                StructuredRequirement(
                    logical_block=logical_block,
                    req_id=req_id,
                    requirement_type=self._requirement_type(req_id, logical_block, purpose, section_type)
                    if requirement_type == "functional"
                    else requirement_type,
                    purpose=purpose,
                    inputs=self._sentence_for(purpose, {"input", "enter", "using", "from", "receive", "provided"}) or "Not specified.",
                    outputs=self._sentence_for(purpose, {"output", "display", "message", "confirmation", "report", "list"}) or "Not specified.",
                    process=self._sentence_for(purpose, {"when", "after", "before", "issue", "return", "update", "track", "manage"}) or "Not specified.",
                    validation=self._sentence_for(purpose, {"within", "authorized", "valid", "validate", "respond", "access"}) or "Not specified.",
                    acceptance_criteria=self._sentence_for(purpose, {"within", "success", "confirmation", "available", "authorized"}) or "Not specified.",
                    external_events=self._sentence_for(purpose, {"trigger", "event", "when"}) or "Not specified.",
                    temporal_events=self._sentence_for(purpose, {"time", "schedule", "periodic"}) or "Not specified.",
                    metadata=metadata,
                    field_evidence=self._evidence_from_fields({"purpose": purpose}, purpose, "implicit-text", metadata.confidence),
                )
            )

        return requirements

    def _extract_labeled_text_blocks(self, parsed: ParsedDocument) -> List[StructuredRequirement]:
        lines = [normalize_space(line) for line in (parsed.text or "").splitlines()]
        requirements: List[StructuredRequirement] = []
        counters: Dict[str, int] = defaultdict(int)
        current_section = "General"
        idx = 0

        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue

            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if FIELD_LABEL_PATTERN.fullmatch(next_line or ""):
                title = self._clean_requirement_candidate(line)
            else:
                if self._is_probable_heading(line):
                    current_section = self._clean_heading_text(line)
                    idx += 1
                    continue
                idx += 1
                continue

            fields: Dict[str, str] = {}
            block_parts = [title]
            idx += 1

            while idx < len(lines):
                label_line = lines[idx]
                label_match = FIELD_LABEL_PATTERN.fullmatch(label_line or "")
                if not label_match:
                    break

                label = self._normalize_label(label_match.group(1))
                key = "purpose" if label == "description" else self._field_key(label)
                idx += 1
                value_parts: List[str] = []

                while idx < len(lines):
                    value_line = lines[idx]
                    if not value_line:
                        idx += 1
                        if value_parts:
                            break
                        continue
                    if FIELD_LABEL_PATTERN.fullmatch(value_line):
                        break
                    if (
                        self._is_probable_heading(value_line)
                        and value_parts
                        and key != "inputs"
                    ):
                        break
                    if (
                        idx + 1 < len(lines)
                        and FIELD_LABEL_PATTERN.fullmatch(lines[idx + 1] or "")
                        and self._normalize_label(lines[idx + 1]) == "description"
                        and value_parts
                    ):
                        break
                    value_parts.append(value_line)
                    block_parts.append(value_line)
                    idx += 1

                value = self._strip_bullet_prefixes("\n".join(value_parts))
                if key and value:
                    fields[key] = value[:1000]

            if not self._looks_like_requirement(fields):
                continue

            requirement_type = self._implicit_requirement_type(current_section, f"{title} {' '.join(fields.values())}")
            req_id = self._generated_req_id(requirement_type, counters)
            fields, inferred_fields = self._prepare_requirement_fields(fields, "\n".join(block_parts), req_id, requirement_type)
            metadata = self._metadata_from_text(
                text="\n".join(block_parts),
                fields=fields,
                logical_block=title or current_section,
                block_index=None,
                method="labeled-text",
                confidence=0.72,
            )
            field_evidence = self._evidence_from_fields(fields, "\n".join(block_parts), "labeled-text", 0.72)
            field_evidence.update(self._evidence_from_inferred_fields(inferred_fields, "\n".join(block_parts)))
            requirements.append(
                StructuredRequirement(
                    logical_block=title or current_section,
                    req_id=req_id,
                    requirement_type=requirement_type,
                    purpose=fields.get("purpose") or title or "Not specified.",
                    inputs=fields.get("inputs") or "Not specified.",
                    outputs=fields.get("outputs") or "Not specified.",
                    process=fields.get("process") or "Not specified.",
                    validation=fields.get("validation") or "Not specified.",
                    acceptance_criteria=fields.get("acceptance_criteria") or "Not specified.",
                    pre_loaded_values=fields.get("pre_loaded_values") or "Not specified.",
                    default_values=fields.get("default_values") or "Not specified.",
                    valid_range_of_values=fields.get("valid_range_of_values") or "Not specified.",
                    data_latency_period=fields.get("data_latency_period") or "Not specified.",
                    data_retention_period=fields.get("data_retention_period") or "Not specified.",
                    data_rate=fields.get("data_rate") or "Not specified.",
                    external_events=fields.get("external_events") or "Not specified.",
                    temporal_events=fields.get("temporal_events") or "Not specified.",
                    constraints=fields.get("constraints") or "Not specified.",
                    effects_on_other_systems=fields.get("effects_on_other_systems") or "Not specified.",
                    assumptions=fields.get("assumptions") or "Not specified.",
                    testability=fields.get("testability") or "Not specified.",
                    metadata=metadata,
                    field_evidence=field_evidence,
                )
            )

        return requirements

    def _iter_text_lines(self, parsed: ParsedDocument) -> Iterable[str]:
        for block in parsed.blocks:
            if block.kind == "table" and block.text:
                yield block.text
                continue
            for line in (block.text or "").splitlines():
                yield line
        if not parsed.blocks:
            for line in (parsed.text or "").splitlines():
                yield line

    def _is_probable_heading(self, line: str) -> bool:
        text = line.strip()
        if not text or len(text) > 120:
            return False
        if REQ_ID_PATTERN.search(text):
            return False
        if BULLET_PREFIX_PATTERN.match(text):
            return False
        lowered = HEADING_NUMBER_PREFIX.sub("", text.rstrip(":").strip()).lower()
        if lowered in {
            "introduction",
            "purpose",
            "scope",
            "users",
            "assumptions",
            "functional requirements",
            "non-functional requirements",
            "system flow",
            "technology stack",
            "sample input",
            "expected output",
            "future enhancements",
            "conclusion",
        }:
            return True
        return bool(HEADING_LIKE_PATTERN.match(text)) and not REQUIREMENT_MODAL_PATTERN.search(text)

    def _clean_heading_text(self, line: str) -> str:
        text = HEADING_NUMBER_PREFIX.sub("", line).strip().strip(":")
        return text or "General"

    def _clean_requirement_candidate(self, line: str) -> str:
        text = BULLET_PREFIX_PATTERN.sub("", line).strip()
        text = re.sub(r"^(?:the\s+)?system\s+(?:will\s+)?allow\s*[:\-]\s*", "", text, flags=re.I)
        text = re.sub(r"^(?:the\s+)?system\s+(?:shall|should|must|will)\s+allow\s*[:\-]\s*", "", text, flags=re.I)
        return normalize_space(text)

    def _is_implicit_requirement_candidate(self, text: str, current_section: str, was_bullet: bool) -> bool:
        if len(text) < 12 or len(text) > 700:
            return False
        if text.rstrip().endswith(":"):
            return False
        lowered = text.lower()
        if "<" in text and ">" in text:
            return False
        if lowered in {"yes", "no", "n/a", "not applicable", "not specified"}:
            return False
        if re.search(r"^(?:frontend|backend|database|framework|component)\s+", lowered):
            return False
        in_requirement_section = bool(REQUIREMENT_SECTION_PATTERN.search(current_section or ""))
        if was_bullet and in_requirement_section:
            return True
        return bool(REQUIREMENT_MODAL_PATTERN.search(text))

    def _requirement_sentence(self, text: str) -> str:
        text = self._strip_bullet_prefixes(text)
        if re.match(r"^(?:admin|user|users|student|students|library staff|staff)\b", text, flags=re.I):
            return text[:1].upper() + text[1:]
        if re.match(r"^(?:add|manage|search|issue|return|track|display|store|validate|update|provide|support)\b", text, flags=re.I):
            return f"The system shall {text[0].lower()}{text[1:]}"
        return text[:1].upper() + text[1:]

    def _implicit_requirement_type(self, section: str, text: str) -> str:
        combined = f"{section} {text}"
        if NON_FUNCTIONAL_PATTERN.search(combined):
            return "non-functional"
        if re.search(r"\b(?:security|authorized|authentication|login|credential)\b", combined, flags=re.I):
            return "security"
        if re.search(r"\b(?:safety|hazard|asil|fail-safe)\b", combined, flags=re.I):
            return "safety"
        return "functional"

    def _generated_req_id(self, requirement_type: str, counters: Dict[str, int]) -> str:
        prefix_by_type = {
            "non-functional": "AUTO-NFR",
            "security": "AUTO-SEC",
            "safety": "AUTO-SAFE",
            "cybersecurity": "AUTO-CYB",
        }
        prefix = prefix_by_type.get(requirement_type, "AUTO-FR")
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]:03d}"

    def _implicit_logical_block(self, section: str, purpose: str, requirement_type: str) -> str:
        cleaned_section = self._clean_logical_block(section, requirement_type)
        if cleaned_section != requirement_type.replace("_", " ").title() and cleaned_section != "General":
            return cleaned_section
        words = purpose.split()
        return " ".join(words[:8]).rstrip(".,;:") or requirement_type.replace("_", " ").title()

    def _fields_from_text_requirement(self, chunk: str) -> Dict[str, str]:
        fields, _ = self._fields_from_semantic_block(chunk)
        return fields

    def _prepare_requirement_fields(
        self,
        fields: Dict[str, str],
        source_text: str,
        req_id: str,
        requirement_type: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        prepared = dict(fields)
        inferred: Dict[str, str] = {}

        purpose = self._best_purpose_text(prepared.get("purpose", ""), source_text, req_id, requirement_type)
        if purpose:
            if compact_inline(purpose).lower() != compact_inline(prepared.get("purpose", "")).lower():
                inferred["purpose"] = purpose
            prepared["purpose"] = purpose

        if prepared.get("process"):
            process = self._clean_process_value(prepared["process"])
            if process:
                prepared["process"] = process
            else:
                prepared.pop("process", None)

        validation = self._clean_validation_value(prepared.get("validation", ""))
        if not validation:
            validation = self._best_sentence_for_field(source_text, "validation")
            if validation:
                inferred["validation"] = validation
        if validation:
            prepared["validation"] = validation

        acceptance = self._clean_acceptance_value(prepared.get("acceptance_criteria", ""))
        if acceptance and self._is_duplicate_content(acceptance, validation):
            acceptance = ""
        if not acceptance:
            acceptance = self._infer_acceptance_criteria(prepared, source_text)
            if acceptance:
                inferred["acceptance_criteria"] = acceptance
        if acceptance:
            prepared["acceptance_criteria"] = acceptance
        else:
            prepared.pop("acceptance_criteria", None)

        for key, value in self._infer_missing_context_fields(prepared, source_text).items():
            if self._is_missing_extracted_value(prepared.get(key)):
                prepared[key] = value
                inferred[key] = value

        return prepared, inferred

    def _evidence_from_inferred_fields(self, fields: Dict[str, str], source_text: str) -> Dict[str, FieldEvidence]:
        return {
            key: FieldEvidence(
                value=value,
                confidence=0.68 if key == "acceptance_criteria" else 0.64,
                source_text=normalize_space(source_text)[:1200],
                extraction_method="inferred-field",
                fallback_reason="Inferred from nearby requirement evidence",
            )
            for key, value in fields.items()
            if value
        }

    def _best_purpose_text(self, explicit_value: str, source_text: str, req_id: str, requirement_type: str) -> str:
        colon_list_purpose = self._purpose_from_colon_list(source_text)
        if colon_list_purpose and (
            not explicit_value or compact_inline(explicit_value).endswith(":") or self._is_field_boundary_sentence(explicit_value)
        ):
            return colon_list_purpose

        source_sentences = self._requirement_sentences(source_text)
        explicit_sentences = self._requirement_sentences(explicit_value)
        candidates = explicit_sentences or source_sentences

        if requirement_type == "non-functional" or re.search(r"\bNFR\b|Non[_ -]?Funct", req_id or "", flags=re.I):
            candidates = list(dict.fromkeys([*explicit_sentences, *source_sentences]))
            nfr_sentences = [
                sentence
                for sentence in candidates
                if self._nonfunctional_sentence_score(sentence) > 0 and not self._is_field_boundary_sentence(sentence)
            ]
            nfr_sentences = sorted(
                enumerate(nfr_sentences),
                key=lambda item: (-self._nonfunctional_sentence_score(item[1]), item[0]),
            )
            selected = [sentence for _, sentence in nfr_sentences[:3]]
            if selected:
                return " ".join(selected)

        scored = [
            (self._purpose_sentence_score(sentence), index, sentence)
            for index, sentence in enumerate(candidates)
            if not self._is_field_boundary_sentence(sentence)
        ]
        scored = [item for item in scored if item[0] > 0]
        if scored:
            return max(scored, key=lambda item: (item[0], -item[1]))[2]
        return normalize_space(explicit_value) or self._sentence_for(source_text, {"shall", "must", "should"}) or ""

    def _purpose_from_colon_list(self, source_text: str) -> str:
        lines = [normalize_space(line) for line in (source_text or "").splitlines() if normalize_space(line)]
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not re.search(r"\bshall\b", lowered) or not line.rstrip().endswith(":"):
                continue
            if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES + ACCEPTANCE_SENTENCE_CUES):
                continue
            items: List[str] = []
            for item in lines[index + 1 : index + 8]:
                item_lower = item.lower().strip(":")
                if REQ_ID_PATTERN.search(item) or self._is_field_boundary_sentence(item):
                    break
                if item_lower in {"input", "output", "example logic", "description", "expected result"}:
                    break
                if item and not item.endswith(":"):
                    items.append(item.rstrip("."))
                if len(items) >= 5:
                    break
            if not items:
                continue

            subject = line.rstrip(":").strip()
            if subject.lower() in {"the system shall", "system shall"}:
                return f"The system shall {self._join_list_items(items, lower_initial=True)}."
            return f"{subject} {self._join_list_items(items, lower_initial=True)}."
        return ""

    def _join_list_items(self, items: List[str], lower_initial: bool = False) -> str:
        cleaned = [normalize_space(item).rstrip(".") for item in items if normalize_space(item)]
        if lower_initial:
            cleaned = [item[:1].lower() + item[1:] if item else item for item in cleaned]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _requirement_sentences(self, text: str) -> List[str]:
        sentences: List[str] = []
        for line in self._semantic_lines(text or ""):
            labeled = self._match_labeled_line(line)
            if labeled:
                key, value = labeled
                if key in {"req_id", "customer_req_id", "milestone", "priority"}:
                    continue
                line = value
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", line):
                sentence = normalize_space(sentence)
                if sentence:
                    sentences.append(sentence)
        return sentences

    def _purpose_sentence_score(self, sentence: str) -> int:
        lowered = sentence.lower()
        if not sentence or self._is_metadata_sentence(sentence):
            return 0
        if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES + ACCEPTANCE_SENTENCE_CUES):
            return 0
        if any(cue in lowered for cue in OUTPUT_SENTENCE_CUES):
            return 0
        if "listed below" in lowered or "include:" in lowered:
            return 0
        score = 0
        if re.search(r"\b(?:shall|must|should|will)\b", lowered):
            score += 3
        if re.search(r"\b(?:hvdcdc|bems|system|software|controller|application|module)\b", lowered):
            score += 3
        if lowered.startswith(("the ", "this requirement")):
            score += 1
        if any(
            cue in lowered
            for cue in (
                "shall support",
                "shall provide",
                "shall detect",
                "shall maintain",
                "shall monitor",
                "shall manage",
                "shall implement",
                "shall acquire",
                "shall disable",
                "shall enable",
                "shall comply",
                "shall not exceed",
            )
        ):
            score += 3
        if "debug pdu" in lowered or re.search(r"\bpdu shall transmit every\b", lowered):
            score -= 5
        return max(score, 0)

    def _nonfunctional_sentence_score(self, sentence: str) -> int:
        lowered = sentence.lower()
        if not re.search(r"\b(?:shall|must|should|will)\b", lowered):
            return 0
        if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES + ACCEPTANCE_SENTENCE_CUES):
            return 0
        score = 0
        for weight, cues in (
            (5, ("startup time", "response time")),
            (4, ("memory utilization", "cpu utilization", "flash memory")),
            (2, ("ambient temperature", "operating temperature")),
            (1, ("performance", "availability", "reliability", "scalability", "maintainability")),
        ):
            if any(cue in lowered for cue in cues):
                score += weight
        return score

    def _is_metadata_sentence(self, sentence: str) -> bool:
        lowered = sentence.lower()
        return any(cue in lowered for cue in METADATA_SENTENCE_CUES)

    def _is_field_boundary_sentence(self, sentence: str) -> bool:
        lowered = sentence.lower()
        return self._is_metadata_sentence(sentence) or any(cue in lowered for cue in FIELD_BOUNDARY_CUES)

    def _clean_process_value(self, value: str) -> str:
        cleaned: List[str] = []
        for line in self._clean_multiline_value(value).splitlines():
            lowered = line.lower()
            if self._is_metadata_sentence(line):
                break
            if any(cue in lowered for cue in FIELD_BOUNDARY_CUES if cue not in {"external event", "temporal event"}):
                break
            if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES + ACCEPTANCE_SENTENCE_CUES):
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _clean_validation_value(self, value: str) -> str:
        cleaned: List[str] = []
        for line in self._clean_multiline_value(value).splitlines():
            lowered = line.lower()
            if self._is_metadata_sentence(line):
                continue
            if any(cue in lowered for cue in ACCEPTANCE_SENTENCE_CUES):
                continue
            if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES):
                cleaned.append(line)
        if cleaned:
            return "\n".join(cleaned).strip()
        value = self._clean_multiline_value(value)
        return value if any(cue in value.lower() for cue in VALIDATION_SENTENCE_CUES) else ""

    def _clean_acceptance_value(self, value: str) -> str:
        cleaned = self._clean_multiline_value(value)
        if not cleaned:
            return ""
        if any(cue in cleaned.lower() for cue in VALIDATION_SENTENCE_CUES) and not any(
            cue in cleaned.lower() for cue in ACCEPTANCE_SENTENCE_CUES
        ):
            return ""
        return cleaned

    def _infer_acceptance_criteria(self, fields: Dict[str, str], source_text: str) -> str:
        purpose = normalize_space(fields.get("purpose") or self._best_purpose_text("", source_text, "", "functional"))
        if not purpose or purpose == "Not specified.":
            return ""
        outputs = normalize_space(fields.get("outputs") or "")
        constraints = normalize_space(fields.get("valid_range_of_values") or fields.get("constraints") or "")
        base = purpose.rstrip(".")
        if len(base) > 220:
            base = base[:217].rstrip(" ,;") + "..."
        parts = [f"Accepted when {base} is demonstrated under the specified operating conditions"]
        if outputs and outputs != "Not specified.":
            output_text = outputs.splitlines()[0].rstrip(".")
            parts.append(f"and the expected output is observed: {output_text}")
        if constraints and constraints != "Not specified.":
            parts.append(f"while satisfying the stated limits: {constraints.splitlines()[0].rstrip('.')}")
        return " ".join(parts) + "."

    def _infer_missing_context_fields(self, fields: Dict[str, str], source_text: str) -> Dict[str, str]:
        inferred: Dict[str, str] = {}
        for key in (
            "data_latency_period",
            "data_retention_period",
            "data_rate",
            "external_events",
            "temporal_events",
            "constraints",
            "effects_on_other_systems",
            "testability",
        ):
            if not self._is_missing_extracted_value(fields.get(key)):
                continue
            value = self._best_sentence_for_field(source_text, key)
            if value:
                inferred[key] = value

        validation = fields.get("validation") or inferred.get("validation") or self._best_sentence_for_field(source_text, "validation")
        if self._is_missing_extracted_value(fields.get("testability")) and validation:
            inferred["testability"] = f"Testable through: {validation}"
        return inferred

    def _best_sentence_for_field(self, source_text: str, field: str) -> str:
        scored: List[Tuple[int, int, str]] = []
        for index, sentence in enumerate(self._requirement_sentences(source_text)):
            score = self._field_sentence_score(sentence, field)
            if score > 0:
                scored.append((score, index, sentence))
        if not scored:
            return ""
        return max(scored, key=lambda item: (item[0], -item[1]))[2]

    def _field_sentence_score(self, sentence: str, field: str) -> int:
        lowered = sentence.lower()
        if not lowered or self._is_metadata_sentence(sentence):
            return 0
        if field == "validation":
            return 4 if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES) else 0
        if field == "data_latency_period":
            score = 0
            if any(cue in lowered for cue in ("latency", "response time", "shall not exceed", "plus or minus")):
                score += 3
            elif "within" in lowered:
                score += 1
            if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds)\b", lowered):
                score += 2
            return score
        if field == "data_retention_period":
            return 4 if any(cue in lowered for cue in ("retention", "retain", "retained", "stored for", "maintained for", "nonvolatile")) else 0
        if field == "data_rate":
            score = 0
            if any(cue in lowered for cue in ("data rate", "frequency", "cycle time", "sampling period", "every ")):
                score += 3
            if re.search(r"\bevery\s+\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds)\b", lowered):
                score += 3
            return score
        if field == "external_events":
            score = 0
            if any(cue in lowered for cue in ("trigger", "whenever", "when ", "ignition on", "command", "request")):
                score += 3
            if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES):
                score -= 3
            return max(score, 0)
        if field == "temporal_events":
            score = 0
            if any(cue in lowered for cue in ("periodic", "scheduled", "time-based", "every ", "after ", "wait", "cycle time")):
                score += 3
            if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds)\b", lowered):
                score += 2
            return score
        if field == "constraints":
            return 3 if any(cue in lowered for cue in ("shall not", "must not", "within", "range", "limited to", "not exceed", "plus or minus")) else 0
        if field == "effects_on_other_systems":
            return 3 if any(cue in lowered for cue in ("broadcast", "transmit", "output", "fault", "state machine", "power module", "other system", "subsystem")) else 0
        if field == "testability":
            return 3 if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES) else 0
        return 0

    def _is_duplicate_content(self, left: str, right: str) -> bool:
        left_norm = re.sub(r"\W+", "", str(left or "").lower())
        right_norm = re.sub(r"\W+", "", str(right or "").lower())
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        return len(left_norm) > 40 and (left_norm in right_norm or right_norm in left_norm)

    def _fields_from_semantic_block(self, chunk: str) -> Tuple[Dict[str, str], Dict[str, FieldEvidence]]:
        collected: Dict[str, List[str]] = defaultdict(list)
        evidence: Dict[str, FieldEvidence] = {}
        lines = self._semantic_lines(chunk)

        def add(key: Optional[str], value: str, source: str, confidence: float, method: str) -> None:
            if not key:
                return
            cleaned = self._clean_multiline_value(value)
            if not cleaned or self._looks_like_template_placeholder(cleaned[:80], {key: cleaned}):
                return
            normalized = compact_inline(cleaned).lower()
            if normalized in {compact_inline(item).lower() for item in collected.get(key, [])}:
                return
            if any(normalized in compact_inline(item).lower() or compact_inline(item).lower() in normalized for item in collected.get(key, [])):
                return
            collected[key].append(cleaned)
            current = evidence.get(key)
            if current is None or confidence >= current.confidence:
                evidence[key] = FieldEvidence(
                    value=cleaned,
                    confidence=round(confidence, 3),
                    source_text=normalize_space(source)[:1200],
                    extraction_method=method,
                )

        for value in SDS_PATTERN.findall(chunk or ""):
            add("customer_req_id", value.upper(), value, 0.94, "metadata-pattern")
            break
        milestone = self._infer_milestone(chunk)
        if milestone:
            add("milestone", milestone, milestone, 0.92, "metadata-pattern")
        priority = self._infer_priority(chunk)
        if priority:
            add("priority", priority, priority, 0.88, "metadata-pattern")

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            labeled = self._match_labeled_line(line)
            key = labeled[0] if labeled else None
            value = labeled[1] if labeled else ""

            if key:
                captured = [value] if value else []
                lookahead = idx + 1
                while lookahead < len(lines):
                    next_line = lines[lookahead]
                    if REQ_ID_PATTERN.search(next_line):
                        break
                    if self._match_labeled_line(next_line):
                        break
                    if self._is_probable_heading(next_line) and captured and not self._is_list_or_step_line(next_line):
                        break
                    if key == "process" and self._is_field_boundary_sentence(next_line):
                        break
                    if key == "validation" and any(cue in next_line.lower() for cue in ACCEPTANCE_SENTENCE_CUES):
                        break
                    if next_line:
                        captured.append(next_line)
                    lookahead += 1
                add(key, "\n".join(captured), "\n".join([line, *captured]), 0.9, "explicit-label-span")
                idx = max(lookahead, idx + 1)
                continue
            idx += 1

        self._capture_multiline_process(lines, add)

        for unit in self._semantic_units(chunk):
            key, confidence = self._classify_field_unit(unit)
            if key:
                add(key, unit, unit, confidence, "semantic-cue")

        fields = {key: self._join_field_values(key, values) for key, values in collected.items() if values}
        fields = self._dedupe_overlapping_fields(fields)
        for key, value in fields.items():
            if key in evidence:
                evidence[key].value = value
            else:
                evidence[key] = FieldEvidence(value=value, confidence=0.6, source_text=value, extraction_method="post-process")
        return fields, evidence

    def _match_labeled_line(self, line: str) -> Optional[Tuple[str, str]]:
        inline = INLINE_FIELD_LABEL_PATTERN.match(line)
        if inline:
            key = self._field_key(self._normalize_label(inline.group(1)))
            return (key, inline.group(2)) if key else None

        if "|" in line:
            parts = [normalize_space(part) for part in line.split("|")]
            if len(parts) >= 2:
                key = self._field_key(self._normalize_label(parts[0]))
                if key:
                    return key, " | ".join(part for part in parts[1:] if part)

        label_match = FIELD_LABEL_PATTERN.fullmatch(line)
        if label_match:
            key = self._field_key(self._normalize_label(label_match.group(1)))
            return (key, "") if key else None
        return None

    def _semantic_lines(self, text: str) -> List[str]:
        lines: List[str] = []
        for raw_line in re.split(r"\n+|(?<=:)\s+(?=(?:Verify|Validate|Enable|Disable|Check|Confirm)\b)", text or ""):
            line = normalize_space(raw_line)
            if line:
                lines.append(line)
        return lines

    def _semantic_units(self, text: str) -> List[str]:
        units: List[str] = []
        for line in self._semantic_lines(text):
            if self._is_list_or_step_line(line):
                units.append(line)
                continue
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
                sentence = normalize_space(sentence)
                if sentence:
                    units.append(sentence)
        return units

    def _classify_field_unit(self, unit: str) -> Tuple[Optional[str], float]:
        lowered = f" {unit.lower()} "
        if self._match_labeled_line(unit) or self._is_metadata_sentence(unit):
            return None, 0.0

        if any(cue in lowered for cue in ACCEPTANCE_SENTENCE_CUES):
            return "acceptance_criteria", 0.86
        if any(cue in lowered for cue in VALIDATION_SENTENCE_CUES):
            return "validation", 0.84
        if "valid range" in lowered or re.search(r"\brange\s+is\b|\bminimum\b|\bmaximum\b", lowered):
            return "valid_range_of_values", 0.78
        if self._field_sentence_score(unit, "data_retention_period") > 0:
            return "data_retention_period", 0.78
        if self._purpose_sentence_score(unit) >= 6:
            return "purpose", 0.8
        if any(cue in lowered for cue in ("process flow", "process requires", "the process requires", "process steps", "steps include", "workflow", "sequence")):
            return "process", 0.82
        if self._field_sentence_score(unit, "data_rate") >= 3:
            return "data_rate", 0.76
        if self._field_sentence_score(unit, "data_latency_period") >= 3:
            return "data_latency_period", 0.76
        if any(cue in lowered for cue in ("inputs required", "input condition", "input required", "receives", "provided by")):
            return "inputs", 0.78
        if any(cue in lowered for cue in OUTPUT_SENTENCE_CUES) or lowered.strip().startswith("the outputs include"):
            return "outputs", 0.8
        if self._field_sentence_score(unit, "external_events") >= 3:
            return "external_events", 0.7
        if self._field_sentence_score(unit, "temporal_events") >= 3:
            return "temporal_events", 0.7
        if self._field_sentence_score(unit, "constraints") >= 3:
            return "constraints", 0.7
        if self._field_sentence_score(unit, "effects_on_other_systems") >= 3:
            return "effects_on_other_systems", 0.68
        if self._purpose_sentence_score(unit) >= 5:
            return "purpose", 0.8

        scores: Dict[str, int] = {field: sum(1 for cue in cues if cue in lowered) for field, cues in FIELD_CUES.items()}
        field, score = max(scores.items(), key=lambda item: item[1])
        if score <= 0:
            return None, 0.0
        if field == "purpose" and any(cue in lowered for cue in ("input", "output", "validation", "process", "default", "range")):
            return None, 0.0
        if field == "process" and not any(cue in lowered for cue in ("process", "workflow", "sequence", "steps", "transition")):
            return None, 0.0
        base = 0.62 + min(score, 3) * 0.08
        return field, min(base, 0.86)

    def _capture_multiline_process(self, lines: List[str], add) -> None:
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if not re.search(r"\bprocess(?:\s+flow|\s+specifies|\s+requires)?\b|\bsequence\b|\bsteps?\b", lowered):
                continue
            captured = [line]
            lookahead = idx + 1
            while lookahead < len(lines):
                next_line = lines[lookahead]
                next_lower = next_line.lower()
                if REQ_ID_PATTERN.search(next_line):
                    break
                if self._match_labeled_line(next_line) or self._is_field_boundary_sentence(next_line):
                    break
                if any(cue in next_lower for cue in VALIDATION_SENTENCE_CUES + ACCEPTANCE_SENTENCE_CUES):
                    break
                if self._is_list_or_step_line(next_line) or re.match(r"^(?:enable|disable|check|confirm|wait|transmit|receive|update|broadcast|clear)\b", next_lower):
                    captured.append(next_line)
                    lookahead += 1
                    continue
                if len(captured) > 1:
                    break
                if not self._is_probable_heading(next_line):
                    captured.append(next_line)
                    lookahead += 1
                    continue
                break
            if len(captured) > 1:
                add("process", "\n".join(captured), "\n".join(captured), 0.91, "multiline-process")

    def _is_list_or_step_line(self, line: str) -> bool:
        return bool(BULLET_PREFIX_PATTERN.match(line) or NUMBERED_STEP_PATTERN.match(line))

    def _clean_multiline_value(self, value: str) -> str:
        cleaned_lines: List[str] = []
        for line in (value or "").splitlines():
            line = normalize_space(line)
            if not line:
                continue
            line = re.sub(r"^\s*(?:[•*+\-–—])\s*", "", line).strip()
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _join_field_values(self, field: str, values: List[str]) -> str:
        if field in {"customer_req_id", "milestone", "priority"}:
            return values[0]
        return "\n".join(value for value in values if value)[:2500]

    def _dedupe_overlapping_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        for primary, secondary in (("process", "inputs"), ("process", "outputs"), ("validation", "process")):
            primary_text = compact_inline(fields.get(primary, "")).lower()
            secondary_text = compact_inline(fields.get(secondary, "")).lower()
            if primary_text and secondary_text and primary_text == secondary_text:
                fields.pop(secondary, None)
        return fields

    def _strip_bullet_prefixes(self, value: str) -> str:
        lines = []
        for line in (value or "").splitlines():
            cleaned = re.sub(r"^\s*(?:[•*-]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)
        return normalize_space("\n".join(lines))

    def _value_from_labeled_row(self, row: List[str], key: str) -> str:
        values: List[str] = []
        for cell in row[1:]:
            value = normalize_space(cell)
            if not value:
                continue
            normalized = self._normalize_label(value)
            if self._is_duplicate_label_cell(normalized, key):
                continue
            values.append(value)
        return values[0] if values else ""

    def _is_duplicate_label_cell(self, normalized: str, key: str) -> bool:
        if normalized in {"term", "description"}:
            return True
        aliases = LABEL_ALIASES.get(key, set())
        if normalized in aliases:
            return True
        return any(normalized.startswith(alias) for alias in aliases if len(alias) > 8)

    def _title_from_text_requirement(self, req_id: str, chunk: str) -> str:
        first_line = next((line.strip() for line in (chunk or "").splitlines() if line.strip()), "")
        match = SIMPLE_REQ_HEADING_PATTERN.match(first_line)
        if match and match.group(1).upper() == req_id.upper():
            return normalize_space(match.group(2))[:120]
        inline = re.match(rf"\s*{re.escape(req_id)}\s+(.+?)(?:\n|Description\b|Input\b|Output\b|$)", chunk or "", flags=re.I | re.S)
        if inline:
            return compact_inline(inline.group(1))[:120]
        return ""

    def _extract_project_name(self, parsed: ParsedDocument) -> str:
        lines = [normalize_space(line) for line in (parsed.text or "").splitlines() if normalize_space(line)]
        for i, line in enumerate(lines[:40]):
            if re.fullmatch(r"software requirement specification(?:\s*\(srs\))?", line, flags=re.I):
                for candidate in lines[i + 1 : i + 6]:
                    lowered = candidate.lower()
                    if (
                        candidate
                        and "<" not in candidate
                        and len(candidate) <= 120
                        and "confidentiality" not in lowered
                        and "copyright" not in lowered
                        and lowered != "contents"
                    ):
                        return candidate

        blocks = [block for block in parsed.blocks[:80] if block.kind in {"paragraph", "heading"} and normalize_space(block.text)]
        for i, block in enumerate(blocks):
            if re.fullmatch(r"software requirement specification(?:\s*\(srs\))?", block.text.strip(), flags=re.I):
                for candidate_block in blocks[i + 1 : i + 8]:
                    candidate = normalize_space(candidate_block.text)
                    lowered = candidate.lower()
                    if (
                        candidate
                        and "<" not in candidate
                        and len(candidate) <= 120
                        and "confidentiality" not in lowered
                        and "copyright" not in lowered
                        and lowered != "contents"
                        and not lowered.startswith("template version")
                        and not (candidate_block.style or "").lower().startswith("toc")
                    ):
                        return candidate
        for pattern in (r"project\s*name\s*[:\-]\s*(.+)", r"project\s*[:\-]\s*(.+)", r"product\s*name\s*[:\-]\s*(.+)"):
            match = re.search(pattern, parsed.text, flags=re.I)
            if match:
                value = normalize_space(match.group(1).splitlines()[0])
                if value and "<" not in value:
                    return value[:120]
        return Path(parsed.source_name).stem.replace("_", " ").strip() or "Unnamed Project"

    def _extract_version(self, text: str) -> str:
        match = VERSION_PATTERN.search(text or "")
        return match.group(1) if match else "1.0"

    def _extract_section_text(self, parsed: ParsedDocument, headings: set[str], max_chars: int = 2200) -> str:
        capturing = False
        start_level: Optional[int] = None
        parts: List[str] = []
        for block in parsed.blocks:
            if block.kind == "heading":
                heading = HEADING_NUMBER_PREFIX.sub("", compact_inline(block.text).lower())
                if capturing and block.heading_level and start_level and block.heading_level <= start_level:
                    break
                if heading in headings:
                    capturing = True
                    start_level = block.heading_level or 9
                    continue
            elif capturing and block.kind in {"paragraph", "table"}:
                text = normalize_space(block.text)
                if text and "<" not in text:
                    parts.append(text)
                    if sum(len(p) for p in parts) >= max_chars:
                        break
        result = normalize_space("\n\n".join(parts))[:max_chars]
        return result or "Not specified."

    def _extract_assumptions(self, parsed: ParsedDocument) -> List[str]:
        text = self._extract_section_text(parsed, {"assumptions", "assumption"}, max_chars=1200)
        if text == "Not specified.":
            return []
        return [item.strip(" -\t") for item in re.split(r"\n|;", text) if item.strip(" -\t")][:20]

    def _extract_open_questions(self, parsed: ParsedDocument) -> List[str]:
        text = self._extract_section_text(parsed, {"queries on requirement", "queries", "open questions"}, max_chars=1200)
        if text == "Not specified.":
            return []
        return [item.strip(" -\t") for item in re.split(r"\n|;", text) if item.strip(" -\t")][:20]

    def _extract_revision_history(self, parsed: ParsedDocument) -> List[RevisionHistoryEntry]:
        approval_rows: List[List[str]] = []
        change_rows: List[List[str]] = []
        generic_entries: List[RevisionHistoryEntry] = []
        current_heading = ""

        for block in parsed.blocks:
            if block.kind == "heading":
                current_heading = self._clean_heading_name(block.text)
                continue
            if block.kind != "table" or not block.rows:
                continue
            headers = [self._normalize_label(cell) for cell in block.rows[0]]
            in_revision_area = "revision approval history" in current_heading or (
                headers and "version" in headers[0] and any("author" in header for header in headers)
            )
            if in_revision_area and any("author" in header for header in headers):
                approval_rows = block.rows
                continue
            if in_revision_area and len(headers) >= 2 and "version" in headers[0] and "description" in headers[1]:
                change_rows = block.rows
                continue
            if not in_revision_area:
                generic_entries.extend(self._revision_entries_from_generic_table(block.rows, current_heading))

        change_by_version: Dict[str, str] = {}
        for row in change_rows[1:]:
            version = normalize_space(row[0] if row else "")
            description = normalize_space(row[1] if len(row) > 1 else "")
            if version and description and not version.lower().startswith("version"):
                change_by_version[version] = description

        entries: List[RevisionHistoryEntry] = []
        for row in approval_rows[1:]:
            cells = [normalize_space(cell) for cell in row]
            if not cells or not cells[0] or cells[0].lower().startswith("version"):
                continue
            version = cells[0]
            if not re.search(r"\b(?:draft|baseline|version|\d+\.\d+)", version, flags=re.I):
                continue
            entry = RevisionHistoryEntry(
                version=version,
                author=self._cell(cells, 1),
                author_date=self._cell(cells, 2),
                reviewer=self._cell(cells, 3),
                reviewer_date=self._cell(cells, 4),
                approver=self._cell(cells, 5),
                approver_date=self._cell(cells, 6),
                change_description=change_by_version.get(version, ""),
                confidence=0.93,
                source_text=" | ".join(cells),
            )
            entries.append(entry)

        if entries:
            return entries
        if generic_entries:
            return self._dedupe_revision_entries(generic_entries)

        text = self._extract_section_text(parsed, {"revision & approval history", "revision approval history"}, max_chars=4000)
        if text == "Not specified.":
            text = self._revision_text_window(parsed.text)
        for line in text.splitlines():
            match = re.match(r"\s*((?:Draft|Baseline|Version)?\s*\d+(?:\.\d+)*)\s+(.+?)\s+(\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4})", line, flags=re.I)
            if match:
                entries.append(
                    RevisionHistoryEntry(
                        version=normalize_space(match.group(1)),
                        author=normalize_space(match.group(2)),
                        author_date=normalize_space(match.group(3)),
                        confidence=0.62,
                        source_text=line,
                    )
                )
        return entries

    def _revision_entries_from_generic_table(self, rows: List[List[str]], current_heading: str) -> List[RevisionHistoryEntry]:
        if len(rows) < 2:
            return []
        heading_hint = "revision" in current_heading or "approval" in current_heading or "change history" in current_heading
        header_index = -1
        headers: List[str] = []
        for index, row in enumerate(rows[:4]):
            normalized = [self._normalize_label(cell) for cell in row]
            header_text = " ".join(normalized)
            has_version = any(self._is_revision_version_header(header) for header in normalized)
            has_revision_context = any(
                cue in header_text
                for cue in ("author", "review", "approv", "date", "description", "change", "summary", "remarks")
            )
            if has_version and (has_revision_context or heading_hint):
                header_index = index
                headers = normalized
                break
        if header_index < 0:
            return []

        version_idx = self._revision_column(headers, ("version", "revision", "rev"))
        description_idx = self._revision_column(headers, ("description", "change", "summary", "remarks", "comment"))
        author_idx = self._revision_column(headers, ("author", "prepared", "created", "modified", "owner"))
        reviewer_idx = self._revision_column(headers, ("reviewer", "reviewed"))
        approver_idx = self._revision_column(headers, ("approver", "approved"))
        date_indexes = [i for i, header in enumerate(headers) if "date" in header]
        if version_idx is None:
            return []
        if description_idx is None and author_idx is None and not date_indexes and not heading_hint:
            return []

        entries: List[RevisionHistoryEntry] = []
        for row in rows[header_index + 1 :]:
            cells = [normalize_space(cell) for cell in row]
            version = self._cell(cells, version_idx)
            if not self._looks_like_revision_version(version):
                continue
            entry = RevisionHistoryEntry(
                version=version,
                author=self._cell(cells, author_idx) if author_idx is not None else "",
                author_date=self._revision_date_for(cells, headers, author_idx, date_indexes, 0),
                reviewer=self._cell(cells, reviewer_idx) if reviewer_idx is not None else "",
                reviewer_date=self._revision_date_for(cells, headers, reviewer_idx, date_indexes, 1),
                approver=self._cell(cells, approver_idx) if approver_idx is not None else "",
                approver_date=self._revision_date_for(cells, headers, approver_idx, date_indexes, 2),
                change_description=self._cell(cells, description_idx) if description_idx is not None else "",
                confidence=0.88,
                source_text=" | ".join(cells),
            )
            if entry.author or entry.author_date or entry.reviewer or entry.change_description:
                entries.append(entry)
        return entries

    def _revision_column(self, headers: List[str], aliases: Tuple[str, ...]) -> Optional[int]:
        for index, header in enumerate(headers):
            if any(alias in header for alias in aliases):
                return index
        return None

    def _is_revision_version_header(self, header: str) -> bool:
        return header in {"version", "revision", "rev"} or header.startswith("version ") or header.startswith("revision ")

    def _looks_like_revision_version(self, value: str) -> bool:
        text = normalize_space(value)
        if not text or text.lower().startswith(("version", "revision", "rev")):
            return False
        return bool(re.search(r"\b(?:draft|baseline|version|rev)?\s*\d+(?:\.\d+)*\b", text, flags=re.I))

    def _revision_date_for(
        self,
        cells: List[str],
        headers: List[str],
        person_index: Optional[int],
        date_indexes: List[int],
        fallback_position: int,
    ) -> str:
        if person_index is not None:
            for index in range(person_index + 1, min(len(headers), person_index + 3)):
                if "date" in headers[index]:
                    return self._cell(cells, index)
        if fallback_position < len(date_indexes):
            return self._cell(cells, date_indexes[fallback_position])
        return ""

    def _revision_text_window(self, text: str) -> str:
        match = re.search(r"(revision\s*(?:&|and)?\s*approval\s*history|revision\s*history|change\s*history)", text or "", flags=re.I)
        if not match:
            return ""
        return (text or "")[match.start() : match.start() + 4000]

    def _dedupe_revision_entries(self, entries: List[RevisionHistoryEntry]) -> List[RevisionHistoryEntry]:
        by_key: Dict[Tuple[str, str, str], RevisionHistoryEntry] = {}
        for entry in entries:
            key = (
                compact_inline(entry.version).lower(),
                compact_inline(entry.author_date).lower(),
                compact_inline(entry.change_description).lower(),
            )
            current = by_key.get(key)
            if current is None or entry.confidence >= current.confidence:
                by_key[key] = entry
        return list(by_key.values())

    def _extract_definitions(self, parsed: ParsedDocument) -> List[DefinitionEntry]:
        entries: List[DefinitionEntry] = []
        current_heading = ""
        for block in parsed.blocks:
            if block.kind == "heading":
                current_heading = self._clean_heading_name(block.text)
                continue
            if block.kind != "table" or not block.rows:
                continue
            headers = [self._normalize_label(cell) for cell in block.rows[0]]
            if "definition" not in current_heading and not (headers and "definition" in headers[0]):
                continue
            for row in block.rows[1:]:
                term = normalize_space(row[0] if row else "")
                description = normalize_space(row[1] if len(row) > 1 else "")
                if term and description:
                    entries.append(DefinitionEntry(term=term, description=description, confidence=0.92, source_text=" | ".join(row)))
            break
        return entries

    def _extract_references(self, parsed: ParsedDocument) -> List[ReferenceEntry]:
        entries: List[ReferenceEntry] = []
        current_heading = ""
        for block in parsed.blocks:
            if block.kind == "heading":
                current_heading = self._clean_heading_name(block.text)
                continue
            if block.kind != "table" or not block.rows:
                continue
            headers = [self._normalize_label(cell) for cell in block.rows[0]]
            if "references" not in current_heading and not (len(headers) >= 2 and "document" in headers[1]):
                continue
            for row in block.rows[1:]:
                cells = [normalize_space(cell) for cell in row]
                if not any(cells):
                    continue
                entries.append(
                    ReferenceEntry(
                        number=self._cell(cells, 0),
                        document=self._cell(cells, 1),
                        version=self._cell(cells, 2),
                        remarks=self._cell(cells, 3),
                        confidence=0.9,
                        source_text=" | ".join(cells),
                    )
                )
            break
        return entries

    def _cell(self, cells: List[str], index: int) -> str:
        return cells[index] if index < len(cells) else ""

    def _metadata_from_text(
        self,
        text: str,
        fields: Dict[str, str],
        logical_block: str,
        block_index: Optional[int],
        method: str,
        confidence: float,
    ) -> RequirementMetadata:
        asil_match = ASIL_PATTERN.search(text) or ASIL_PATTERN.search(fields.get("asil", ""))
        cal_match = CAL_PATTERN.search(text) or CAL_PATTERN.search(fields.get("cal", ""))
        sds_references = list(dict.fromkeys(match.upper() for match in SDS_PATTERN.findall(text)))
        return RequirementMetadata(
            customer_req_id=fields.get("customer_req_id") or (sds_references[0] if sds_references else None),
            milestone=fields.get("milestone") or self._infer_milestone(text),
            priority=fields.get("priority") or self._infer_priority(text),
            asil=asil_match.group(0).upper().replace(" ", "") if asil_match else None,
            cal=cal_match.group(0).upper().replace(" ", "") if cal_match else None,
            can_ids=list(dict.fromkeys(CAN_ID_PATTERN.findall(text))),
            sds_references=sds_references,
            source_section=logical_block,
            source_block_index=block_index,
            extraction_method=method,
            evidence=normalize_space(text[:1200]),
            confidence=confidence,
        )

    def _infer_milestone(self, text: str) -> Optional[str]:
        match = MILESTONE_PATTERN.search(text or "")
        if match:
            return match.group(1).strip().rstrip(".,;")
        direct = re.search(r"\b(?:A-?Sample|B-?Sample|C-?Sample)?/?PF\d+(?:\.\d+)?\b", text or "", flags=re.I)
        return direct.group(0).strip().rstrip(".,;") if direct else None

    def _infer_priority(self, text: str) -> Optional[str]:
        match = PRIORITY_PATTERN.search(text or "")
        if match:
            return match.group(1).strip().title()
        direct = re.search(r"\b(Urgent|High|Medium|Low)\s+priority\b", text or "", flags=re.I)
        return direct.group(1).strip().title() if direct else None

    def _evidence_from_fields(
        self,
        fields: Dict[str, str],
        source_text: str,
        method: str,
        confidence: float,
    ) -> Dict[str, FieldEvidence]:
        evidence: Dict[str, FieldEvidence] = {}
        for field, value in fields.items():
            if value in (None, "", "Not specified."):
                continue
            evidence[field] = FieldEvidence(
                value=str(value),
                confidence=round(confidence, 3),
                source_text=normalize_space(source_text)[:1200],
                extraction_method=method,
            )
        return evidence

    def _normalize_label(self, label: str) -> str:
        label = compact_inline(label).lower()
        label = label.replace("/", " ")
        label = re.sub(r"[^a-z0-9() ]+", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label

    def _clean_heading_name(self, value: str) -> str:
        return HEADING_NUMBER_PREFIX.sub("", self._normalize_label(value)).strip()

    def _field_key(self, label: str) -> Optional[str]:
        if not label:
            return None
        if label == "requirement":
            return "purpose"
        if label.startswith("testability with respect"):
            return "testability"
        if label.startswith("feasible within project constraints"):
            return "feasible"
        if label.startswith("critical"):
            return "critical"
        for key, aliases in LABEL_ALIASES.items():
            if label in aliases:
                return key
            if any(alias in label for alias in aliases if len(alias) > 8):
                return key
        return None

    def _looks_like_requirement(self, fields: Dict[str, str]) -> bool:
        meaningful = {"purpose", "inputs", "outputs", "process", "validation", "acceptance_criteria"}
        return len(meaningful.intersection(fields)) >= 1

    def _looks_like_template_placeholder(self, req_id: str, fields: Dict[str, str]) -> bool:
        values = " ".join(fields.values()).lower()
        placeholder_hits = values.count("<") + values.count(">")
        generic_ids = {"rq01", "rq0n", "requirement description", "cs-req-001", "<cs-req-001>"}
        if req_id.strip().lower() in generic_ids and placeholder_hits:
            return True
        template_phrases = (
            "purpose of the functionality",
            "complete transaction process",
            "map derived requirements",
            "list out all validation rules",
            "mention safety requirement id",
        )
        return any(phrase in values for phrase in template_phrases)

    def _requirement_type(self, req_id: str, logical_block: str, text: str, section_type: str) -> str:
        combined = f"{req_id} {logical_block} {text}".lower()
        if "cyber" in combined or "cs_req" in combined or "cs-req" in combined or section_type == "cybersecurity":
            return "cybersecurity"
        if re.search(r"\bsafe\b|_safe_|-safe-", req_id, flags=re.I) or "safety" in combined or "safreq" in combined or "asil" in combined or section_type == "safety":
            return "safety"
        if re.search(r"\bnfr\b|_nfr_|-nfr-|\bnon[_ -]?funct", req_id, flags=re.I) or NON_FUNCTIONAL_PATTERN.search(combined):
            return "non-functional"
        if re.search(r"\bcomm\b|_comm_|-comm-", req_id, flags=re.I) or section_type == "communication":
            return "interface"
        if re.search(r"\bdiag\b|_diag_|-diag-|\bobd\b|_obd_|-obd-", req_id, flags=re.I) or "dtc" in combined or section_type == "diagnostics":
            return "diagnostic"
        if "interface" in combined or section_type in {"communication", "interfaces"}:
            return "interface" if section_type == "interfaces" else "functional"
        return "functional"

    def _infer_logical_block(self, req_id: str, chunk: str) -> str:
        text = f"{req_id} {chunk}"
        section = self.classifier.classify(text)
        if section == "general":
            return "General"
        return section.replace("_", " ").title()

    def _clean_logical_block(self, logical_block: str, requirement_type: str) -> str:
        text = HEADING_NUMBER_PREFIX.sub("", logical_block or "").strip()
        if not text or text.lower() in {"requirements", "safety requirements", "business rules", "screens / wireframes"}:
            return requirement_type.replace("_", " ").title()
        return text

    def _sentence_for(self, text: str, keywords: set[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                return normalize_space(sentence)[:700]
        return ""

    def _is_template_heading(self, heading: str) -> bool:
        return "<" in heading or heading.lower() in {"requirements", "safety requirements", "business rules", "screen(s)/ wireframes", "screens / wireframes"}

    def _dedupe_requirements(self, requirements: Iterable[StructuredRequirement]) -> List[StructuredRequirement]:
        by_id: Dict[str, StructuredRequirement] = {}
        for req in requirements:
            key = req.req_id.upper()
            current = by_id.get(key)
            if current is None or req.metadata.confidence >= current.metadata.confidence:
                by_id[key] = req
        return list(by_id.values())

    def _merge_requirement_sources(
        self,
        table_requirements: Iterable[StructuredRequirement],
        text_requirements: Iterable[StructuredRequirement],
    ) -> List[StructuredRequirement]:
        by_id: Dict[str, StructuredRequirement] = {}
        for req in table_requirements:
            by_id[req.req_id.upper()] = req

        for text_req in text_requirements:
            key = text_req.req_id.upper()
            existing = by_id.get(key)
            if existing is None:
                by_id[key] = text_req
                continue
            self._merge_requirement(existing, text_req)

        return list(by_id.values())

    def _merge_requirement(self, target: StructuredRequirement, source: StructuredRequirement) -> None:
        merge_fields = (
            "purpose",
            "inputs",
            "outputs",
            "process",
            "validation",
            "acceptance_criteria",
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
            "testability",
            "status",
            "comments",
        )
        for field in merge_fields:
            current_value = getattr(target, field, "")
            source_value = getattr(source, field, "")
            if self._is_missing_extracted_value(current_value) and not self._is_missing_extracted_value(source_value):
                setattr(target, field, source_value)
                if field in source.field_evidence:
                    target.field_evidence[field] = source.field_evidence[field]

        for metadata_field in ("customer_req_id", "milestone", "priority", "asil", "cal"):
            current_value = getattr(target.metadata, metadata_field, None)
            source_value = getattr(source.metadata, metadata_field, None)
            if self._is_missing_extracted_value(current_value) and not self._is_missing_extracted_value(source_value):
                setattr(target.metadata, metadata_field, source_value)

        target.metadata.sds_references = list(dict.fromkeys([*target.metadata.sds_references, *source.metadata.sds_references]))
        target.metadata.can_ids = list(dict.fromkeys([*target.metadata.can_ids, *source.metadata.can_ids]))
        target.metadata.confidence = max(float(target.metadata.confidence or 0.0), float(source.metadata.confidence or 0.0))
        target.field_evidence.update({key: value for key, value in source.field_evidence.items() if key not in target.field_evidence})

    def _is_missing_extracted_value(self, value: Any) -> bool:
        text = str(value or "").strip()
        return not text or text in {"Not specified.", "No derived requirement identified.", "No specific access restrictions identified."}

    def _project_confidence(self, project: SRSProject) -> float:
        if not project.requirements:
            return 0.0
        scores = [req.metadata.confidence for req in project.requirements]
        completeness = mean(
            1.0
            if all(getattr(req, field) != "Not specified." for field in ("purpose", "inputs", "outputs", "process"))
            else 0.65
            for req in project.requirements
        )
        return round((mean(scores) * 0.7) + (completeness * 0.3), 3)


class SRSIntelligencePipeline:
    def __init__(self, parser: DocumentParser | None = None, extractor: RequirementExtractor | None = None) -> None:
        self.parser = parser or DocumentParser()
        self.extractor = extractor or RequirementExtractor()

    def run_path(self, path: str | Path) -> SRSProject:
        return self.extractor.extract(self.parser.parse_path(path))

    def run_bytes(self, file_bytes: bytes, source_name: str) -> SRSProject:
        return self.extractor.extract(self.parser.parse_bytes(file_bytes, source_name))
