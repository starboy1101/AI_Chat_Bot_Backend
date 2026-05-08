from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from srs_generator.models import ParsedDocument, RequirementMetadata, SRSProject, StructuredRequirement
from srs_generator.parser import DocumentParser
from srs_generator.utils import compact_inline, normalize_space, truthy_text
from srs_generator.validator import SRSValidator


REQ_ID_PATTERN = re.compile(
    r"\b(?:SWRS|SWR|REQ|RQ|SAFREQ|SW_SAF_REQ|CS_REQ|CS-REQ)[A-Z0-9_-]*[_-]?\d+[A-Z0-9_-]*\b",
    re.I,
)
SDS_PATTERN = re.compile(r"\bSDS[-_A-Z0-9]*-\d+\b", re.I)
CAN_ID_PATTERN = re.compile(r"\b(?:0x[0-9A-Fa-f]{2,8}|CAN\s*ID\s*[:#-]?\s*[0-9A-Fa-fx]+)\b")
ASIL_PATTERN = re.compile(r"\bASIL\s*(?:QM|[ABCD])\b", re.I)
CAL_PATTERN = re.compile(r"\bCAL\s*[1-4]\b", re.I)
VERSION_PATTERN = re.compile(r"\b(?:version|baseline|draft)\s*[: ]\s*([0-9]+(?:\.[0-9]+)*)\b", re.I)
HEADING_NUMBER_PREFIX = re.compile(r"^\d+(?:\.\d+)*\.?\s*")

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
    "purpose": {"purpose", "purpose1", "description", "detailed requirements specification", "detailed requirement specification"},
    "derived_requirement": {"derived requirement", "derived from"},
    "priority": {"requirement priority", "priority"},
    "access_restrictions": {"access restrictions", "access restriction"},
    "inputs": {"inputs", "input(s)", "input"},
    "outputs": {"outputs", "output(s)", "output"},
    "process": {"process", "workflow", "sequence"},
    "mandatory_fields": {"mandatory fields", "mandatory field"},
    "validation": {"validation rules verification criteria", "validation rules", "verification criteria", "verification method", "testability with respect to test environment yes/no"},
    "constraints": {"constraints", "constraint"},
    "assumptions": {"assumptions", "assumption"},
    "failure_scenario": {"failure scenario", "fault scenario"},
    "action_on_failure": {"action if failure", "actions if failure", "action on failure"},
    "acceptance_criteria": {"acceptance criteria", "requirement acceptance criteria"},
    "status": {"requirement status", "status"},
    "critical": {"critical"},
    "feasible": {"feasible"},
    "comments": {"comments", "remarks", "rationale"},
    "asil": {"asil level", "asil"},
    "cal": {"cal level applicable", "cal"},
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
        project = SRSProject(
            project_name=self._extract_project_name(parsed),
            version=self._extract_version(parsed.text),
            source_name=parsed.source_name,
            system_overview=self._extract_section_text(parsed, {"overview", "system overview", "software overview"}),
            scope=self._extract_section_text(parsed, {"scope"}),
            purpose=self._extract_section_text(parsed, {"purpose of the document", "purpose", "introduction"}),
            operating_environment=self._extract_section_text(parsed, {"operating environment"}),
            acceptance_criteria=self._extract_section_text(parsed, {"overall acceptance criteria", "acceptance criteria"}),
            diagnostics=self._extract_section_text(parsed, {"diagnostics", "general diagnostics"}),
            interfaces={
                "user": self._extract_section_text(parsed, {"user interface", "external interfaces"}),
                "hardware": self._extract_section_text(parsed, {"hardware interface", "external interfaces"}),
                "software": self._extract_section_text(parsed, {"software interface", "external interfaces"}),
            },
        )

        requirements = self._extract_table_requirements(parsed)
        if not requirements:
            requirements = self._extract_text_requirements(parsed)

        project.requirements = self._dedupe_requirements(requirements)
        project.safety_requirements = [r for r in project.requirements if r.requirement_type == "safety"]
        project.cybersecurity_requirements = [r for r in project.requirements if r.requirement_type == "cybersecurity"]
        project.assumptions = self._extract_assumptions(parsed)
        project.open_questions = self._extract_open_questions(parsed)
        project.confidence = self._project_confidence(project)
        project.validation_findings = SRSValidator().validate(project)
        return project

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
            req = self._requirement_from_table(block.rows, current_heading, current_section_type, block.index)
            if req:
                requirements.append(req)
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
            value = normalize_space(row[1])
            if value:
                fields[key] = value

        req_id = fields.get("req_id")
        if not req_id or not self._looks_like_requirement(fields):
            return None
        if self._looks_like_template_placeholder(req_id, fields):
            return None

        evidence = "\n".join(" | ".join(cell for cell in row if cell) for row in rows[:12])
        requirement_type = self._requirement_type(req_id, logical_block, evidence, section_type)
        metadata = self._metadata_from_text(
            text=evidence,
            fields=fields,
            logical_block=logical_block,
            block_index=block_index,
            method="table",
            confidence=0.92,
        )

        return StructuredRequirement(
            logical_block=self._clean_logical_block(logical_block, requirement_type),
            req_id=req_id.strip(),
            requirement_type=requirement_type,
            purpose=fields.get("purpose") or "Not specified.",
            inputs=fields.get("inputs") or "Not specified.",
            outputs=fields.get("outputs") or "Not specified.",
            process=fields.get("process") or "Not specified.",
            validation=fields.get("validation") or "Not specified.",
            acceptance_criteria=fields.get("acceptance_criteria") or fields.get("validation") or "Not specified.",
            derived_requirement=fields.get("derived_requirement") or "No derived requirement identified.",
            access_restrictions=fields.get("access_restrictions") or "No specific access restrictions identified.",
            mandatory_fields=fields.get("mandatory_fields") or "Not specified.",
            constraints=fields.get("constraints") or "Not specified.",
            assumptions=fields.get("assumptions") or "Not specified.",
            failure_scenario=fields.get("failure_scenario") or "Not specified.",
            action_on_failure=fields.get("action_on_failure") or "Not specified.",
            status=fields.get("status") or "Proposed",
            critical=truthy_text(fields.get("critical")),
            feasible=truthy_text(fields.get("feasible")),
            comments=fields.get("comments") or "",
            metadata=metadata,
        )

    def _extract_text_requirements(self, parsed: ParsedDocument) -> List[StructuredRequirement]:
        requirements: List[StructuredRequirement] = []
        matches = list(REQ_ID_PATTERN.finditer(parsed.text))
        if not matches:
            return requirements

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(parsed.text), match.end() + 2400)
            chunk = normalize_space(parsed.text[start:end])
            if len(chunk) < 30:
                continue
            req_id = match.group(0).strip()
            if self._looks_like_template_placeholder(req_id, {"text": chunk}):
                continue
            logical_block = self._infer_logical_block(req_id, chunk)
            requirement_type = self._requirement_type(req_id, logical_block, chunk, self.classifier.classify(chunk))
            metadata = self._metadata_from_text(
                text=chunk,
                fields={},
                logical_block=logical_block,
                block_index=None,
                method="text",
                confidence=0.68,
            )
            requirements.append(
                StructuredRequirement(
                    logical_block=logical_block,
                    req_id=req_id,
                    requirement_type=requirement_type,
                    purpose=self._sentence_for(chunk, {"shall", "must", "should", "purpose"}) or chunk[:500],
                    inputs=self._sentence_for(chunk, {"input", "receive", "provided", "trigger"}) or "Not specified.",
                    outputs=self._sentence_for(chunk, {"output", "transmit", "send", "store", "display"}) or "Not specified.",
                    process=self._sentence_for(chunk, {"process", "sequence", "when", "after", "before"}) or "Not specified.",
                    validation=self._sentence_for(chunk, {"verify", "validate", "test", "criteria"}) or "Not specified.",
                    acceptance_criteria=self._sentence_for(chunk, {"acceptance", "pass", "criteria", "verify"}) or "Not specified.",
                    critical=truthy_text("critical" if "critical" in chunk.lower() else ""),
                    feasible=truthy_text("yes" if "feasible" in chunk.lower() else ""),
                    metadata=metadata,
                )
            )
        return requirements

    def _extract_project_name(self, parsed: ParsedDocument) -> str:
        blocks = [block for block in parsed.blocks[:80] if block.kind in {"paragraph", "heading"} and normalize_space(block.text)]
        for i, block in enumerate(blocks):
            if re.fullmatch(r"software requirement specification", block.text.strip(), flags=re.I):
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
        return RequirementMetadata(
            customer_req_id=fields.get("customer_req_id"),
            milestone=fields.get("milestone"),
            priority=fields.get("priority"),
            asil=asil_match.group(0).upper().replace(" ", "") if asil_match else None,
            cal=cal_match.group(0).upper().replace(" ", "") if cal_match else None,
            can_ids=list(dict.fromkeys(CAN_ID_PATTERN.findall(text))),
            sds_references=list(dict.fromkeys(match.upper() for match in SDS_PATTERN.findall(text))),
            source_section=logical_block,
            source_block_index=block_index,
            extraction_method=method,
            evidence=normalize_space(text[:1200]),
            confidence=confidence,
        )

    def _normalize_label(self, label: str) -> str:
        label = compact_inline(label).lower()
        label = label.replace("/", " ")
        label = re.sub(r"[^a-z0-9() ]+", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label

    def _field_key(self, label: str) -> Optional[str]:
        if not label:
            return None
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
        if "safety" in combined or "safreq" in combined or "asil" in combined or section_type == "safety":
            return "safety"
        if "diagnostic" in combined or "diag" in combined or "dtc" in combined or section_type == "diagnostics":
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
