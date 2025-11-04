FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip & install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# ✅ Preload SentenceTransformer model
RUN python3 - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ SentenceTransformer model downloaded successfully!")
PY

# ✅ Preload Mistral GGUF model
RUN python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Omkar1803/mistral-7b-gguf",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="/app/cache/models"
)
print("✅ GGUF model downloaded successfully!")
PY

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
