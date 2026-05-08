from io import BytesIO
import pickle
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.core.config import DATA_DIR, FAISS_DIR, PDF_MAX_PAGES
from app.core.device import sentence_transformer_device


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text = ""
        reader = PdfReader(str(path))
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk(text: str, size: int = 400) -> list[str]:
    words, chunks, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if len(cur) >= size:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


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


def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", device=sentence_transformer_device())


def build_database(data_dir: str | Path = DATA_DIR, faiss_dir: str | Path = FAISS_DIR) -> None:
    data_path = Path(data_dir)
    faiss_path = Path(faiss_dir)

    all_chunks: list[str] = []
    all_sources: list[str] = []

    files = sorted(path for path in data_path.iterdir() if path.is_file())
    for path in files:
        text = load_text(path)
        chunks = chunk(text)
        all_chunks.extend(chunks)
        all_sources.extend([path.name] * len(chunks))

    print(f"Found {len(all_chunks)} chunks from {len(files)} files.")

    model = _load_embedding_model()
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors.")

    faiss_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path / "index.faiss"))

    with open(faiss_path / "metadata.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    print(f"FAISS database built and stored in {faiss_path}")


if __name__ == "__main__":
    build_database()
