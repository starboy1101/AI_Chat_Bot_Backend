FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"

RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/hf_home /tmp/st_cache /tmp/hf_hub && chmod -R 777 /tmp
ENV HF_HOME=/tmp/hf_home
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache
ENV HF_HUB_CACHE=/tmp/hf_hub

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
