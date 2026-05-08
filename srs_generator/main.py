from __future__ import annotations

import argparse
from pathlib import Path

from srs_generator.extractor import SRSIntelligencePipeline
from srs_generator.template_engine import SrsTemplateRenderer
from srs_generator.utils import safe_filename, write_json


def build_srs(
    input_path: str | Path,
    template_path: str | Path = "srs_generator/templates/srs_template.docx",
    output_dir: str | Path = "srs_generator/output_docs",
    json_dir: str | Path = "srs_generator/extracted_json",
) -> tuple[Path, Path]:
    project = SRSIntelligencePipeline().run_path(input_path)
    base_name = safe_filename(project.project_name)
    json_path = Path(json_dir) / f"{base_name}.json"
    output_path = Path(output_dir) / f"{base_name}_SRS.docx"
    write_json(json_path, project.to_dict())
    SrsTemplateRenderer(template_path).render(project, output_path)
    return output_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an automotive SRS DOCX from an input engineering document.")
    parser.add_argument("input", help="Input PDF, DOCX, DOC, or TXT document")
    parser.add_argument("--template", default="srs_generator/templates/srs_template.docx", help="Master SRS DOCX template")
    parser.add_argument("--output-dir", default="srs_generator/output_docs", help="Generated DOCX output directory")
    parser.add_argument("--json-dir", default="srs_generator/extracted_json", help="Structured JSON output directory")
    args = parser.parse_args()

    docx_path, json_path = build_srs(args.input, args.template, args.output_dir, args.json_dir)
    print(f"SRS DOCX: {docx_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

