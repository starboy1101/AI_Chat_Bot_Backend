# ✅ Use modern, lightweight image (Python 3.10 + build tools)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
# Avoid source compilation for llama-cpp-python
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"

# Install essential system libs for FAISS / Llama-CPP
RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip and prefer binary wheels
RUN python3 -m pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
