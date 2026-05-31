FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for mounting
RUN mkdir -p /models /vllm-cache /repos

# Copy app code
COPY . .

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
