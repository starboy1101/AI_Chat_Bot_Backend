FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
