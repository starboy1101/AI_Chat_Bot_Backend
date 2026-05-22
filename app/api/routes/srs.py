from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from srs_generator.extractor import SRSIntelligencePipeline
from srs_generator.template_engine import SrsTemplateRenderer
from srs_generator.utils import safe_filename, write_json
from srs_generator.validator import fill_missing_srs_fields

router = APIRouter(prefix="/srs", tags=["srs"])

DEFAULT_TEMPLATE = Path("srs_generator/templates/srs_template.docx")
DEFAULT_OUTPUT_DIR = Path("srs_generator/output_docs")
DEFAULT_JSON_DIR = Path("srs_generator/extracted_json")


def _missing_questions(project) -> list[dict[str, str]]:
    """Build clarification prompts for missing SRS fields.

    This is intentionally dormant for the current document-generation flow:
    production DOCX output must be generated without asking the user follow-up
    questions, and missing values are rendered as "Yet to be filled". Keep this
    helper available for a future interactive review mode.
    """
    return []

    # Future interactive mode can be restored here with:
    # missing field name, field definition/purpose, and why it is needed.


@router.post("/extract")
async def extract_srs_json(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        project = SRSIntelligencePipeline().run_bytes(file_bytes, file.filename or "uploaded_document")
        project = fill_missing_srs_fields(project)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to extract SRS requirements: {exc}") from exc

    return JSONResponse(project.to_dict())


@router.get("/output/{filename}")
async def download_generated_srs(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid output filename.")

    output_path = DEFAULT_OUTPUT_DIR / filename
    if not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Generated SRS document not found.")

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_path.name,
    )


@router.post("/generate")
async def generate_srs_docx(
    file: UploadFile = File(...),
    return_json: bool = Query(False, description="Return structured JSON instead of a DOCX file."),
    allow_partial: bool = Query(True, description="Deprecated; DOCX generation now always proceeds with missing fields filled."),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        project = SRSIntelligencePipeline().run_bytes(file_bytes, file.filename or "uploaded_document")
        project = fill_missing_srs_fields(project)
        base_name = safe_filename(project.project_name)
        json_path = DEFAULT_JSON_DIR / f"{base_name}.json"
        output_path = DEFAULT_OUTPUT_DIR / f"{base_name}_SRS.docx"
        write_json(json_path, project.to_dict())
        SrsTemplateRenderer(DEFAULT_TEMPLATE).render(project, output_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to generate SRS document: {exc}") from exc

    if return_json:
        payload = project.to_dict()
        payload["generated_docx"] = str(output_path)
        payload["extracted_json"] = str(json_path)
        return JSONResponse(payload)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_path.name,
    )
