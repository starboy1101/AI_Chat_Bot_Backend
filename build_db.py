# build_db.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: load text
def load_text(path):
    if path.endswith(".pdf"):
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

# Helper: chunk text
def chunk(text, size=400):
    words, chunks, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if len(cur) >= size:
            chunks.append(" ".join(cur))
            cur = []
    if cur: chunks.append(" ".join(cur))
    return chunks

# Collect all chunks
all_chunks = []
all_sources = []

for fname in os.listdir("data"):
    path = os.path.join("data", fname)
    text = load_text(path)
    chunks = chunk(text)
    all_chunks.extend(chunks)
    all_sources.extend([fname]*len(chunks))

print(f"Found {len(all_chunks)} chunks from {len(os.listdir('data'))} files.")

# Encode all chunks
embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = embeddings.astype("float32")  # FAISS requires float32

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# Save index and metadata
os.makedirs("faiss_db", exist_ok=True)
faiss.write_index(index, "faiss_db/index.faiss")

with open("faiss_db/metadata.pkl", "wb") as f:
    pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

print("✅ FAISS database built and stored in ./faiss_db")
