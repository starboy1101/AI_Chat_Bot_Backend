from io import BytesIO

from pypdf import PdfReader

from app.core.config import PDF_MAX_PAGES


class PDFParseError(Exception):
    pass


def _looks_like_pdf(pdf_bytes: bytes) -> bool:
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        return False
    header_window = bytes(pdf_bytes[:1024])
    header_window = header_window.lstrip(b"\x00\x09\x0a\x0c\x0d\x20")
    return header_window.startswith(b"%PDF-") or (b"%PDF-" in header_window)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not _looks_like_pdf(pdf_bytes):
        raise PDFParseError("Uploaded content is not a valid PDF.")

    try:
        text = read_pdf_text_from_bytes(pdf_bytes, max_pages=PDF_MAX_PAGES)
        if not text or len(text.strip()) < 100:
            raise PDFParseError("PDF unreadable or empty")
        return text
    except Exception as e:
        raise PDFParseError(str(e))


def read_pdf_text_from_bytes(pdf_bytes: bytes, max_pages: int | None = PDF_MAX_PAGES) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    text_parts = []

    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        try:
            extracted = page.extract_text()
        except Exception:
            extracted = None
        if extracted and extracted.strip():
            text_parts.append(extracted.strip())

    return "\n\n".join(text_parts)
