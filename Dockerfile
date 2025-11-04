FROM python:3.10-slim
FROM huggingface/transformers-pytorch-cpu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip separately
RUN python3 -m pip install --upgrade pip

# Install dependencies one by one (avoids broken pipes)
RUN pip install --no-cache-dir -r requirements.txt


# Expose port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
