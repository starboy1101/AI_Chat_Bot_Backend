from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from srs_generator.extractor import SRSIntelligencePipeline
from srs_generator.section_instructions import build_project_field_question
from srs_generator.template_engine import SrsTemplateRenderer
from srs_generator.utils import safe_filename, write_json

router = APIRouter(prefix="/srs", tags=["srs"])

DEFAULT_TEMPLATE = Path("srs_generator/templates/srs_template.docx")
DEFAULT_OUTPUT_DIR = Path("srs_generator/output_docs")
DEFAULT_JSON_DIR = Path("srs_generator/extracted_json")


def _missing_questions(project) -> list[dict[str, str]]:
    questions = []
    for finding in project.validation_findings:
        if finding.code not in {"MISSING_REQUIREMENT_FIELD", "MISSING_ASIL", "MISSING_PROJECT_FIELD"}:
            continue
        if not finding.field:
            continue
        label = finding.field.replace("_", " ").title()
        if finding.code == "MISSING_PROJECT_FIELD":
            questions.append(
                {
                    "target": "project",
                    "field": finding.field,
                    "question": build_project_field_question(finding.field, project),
                }
            )
            continue
        questions.append(
            {
                "target": "requirement",
                "req_id": finding.req_id,
                "field": finding.field,
                "question": f"Please provide {label} for {finding.req_id}.",
            }
        )
    return questions


@router.post("/extract")
async def extract_srs_json(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        project = SRSIntelligencePipeline().run_bytes(file_bytes, file.filename or "uploaded_document")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to extract SRS requirements: {exc}") from exc

    return JSONResponse(project.to_dict())


@router.post("/generate")
async def generate_srs_docx(
    file: UploadFile = File(...),
    return_json: bool = Query(False, description="Return structured JSON instead of a DOCX file."),
    allow_partial: bool = Query(False, description="Generate a DOCX even when mandatory SRS fields are missing."),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        project = SRSIntelligencePipeline().run_bytes(file_bytes, file.filename or "uploaded_document")
        missing_questions = _missing_questions(project)
        base_name = safe_filename(project.project_name)
        json_path = DEFAULT_JSON_DIR / f"{base_name}.json"
        output_path = DEFAULT_OUTPUT_DIR / f"{base_name}_SRS.docx"
        write_json(json_path, project.to_dict())
        if missing_questions and not allow_partial:
            payload = project.to_dict()
            payload["extracted_json"] = str(json_path)
            payload["missing_questions"] = missing_questions
            payload["message"] = "Mandatory SRS fields are missing. Ask these questions before final DOCX generation."
            return JSONResponse(payload, status_code=409)
        SrsTemplateRenderer(DEFAULT_TEMPLATE).render(project, output_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to generate SRS document: {exc}") from exc

    if return_json:
        payload = project.to_dict()
        payload["generated_docx"] = str(output_path)
        payload["extracted_json"] = str(json_path)
        payload["missing_questions"] = missing_questions
        return JSONResponse(payload)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_path.name,
    )
