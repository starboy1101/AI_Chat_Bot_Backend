# ✅ Use modern Python with build tools
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"
# Set cache dirs to /data to avoid permission errors
ENV HF_HOME=/data
ENV TRANSFORMERS_CACHE=/data/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/data/sentence_transformers

# Install required system libs
RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY . .

# Upgrade pip and install deps
RUN python3 -m pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# ✅ Pre-download models at build time
RUN python3 -m sentence_transformers import_from_hub all-MiniLM-L6-v2
RUN python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Omkar1803/mistral-7b-gguf",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="/data/models"
)
PY

# Expose FastAPI port
EXPOSE 7860

# Start backend
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
