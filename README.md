# vLLM Container Manager

A premium Web UI dashboard and FastAPI backend built to configure, monitor, benchmark, and autotune your `vLLM` Docker containers on consumer GPUs.

## Features

- **Docker Container Life-Cycle Management**: Start, stop, and monitor your vLLM server container directly from the web interface.
- **Dynamic Configuration Management**: Parse and modify `run_vllm.sh` arguments on the fly (model, served name, max model length, KV cache settings, dtype, speculative decoding config, etc.).
- **Auto-Tuning Engine**: Scan GPU concurrency metrics to automatically find the optimal `gpu-memory-utilization` that prevents out-of-memory (OOM) situations and host RAM spilling on Windows WDDM/WSL2.
- **Context Capacity Verification**: Run automated prefill scanners that verify model context stability up to physical limits.
- **Real-Time Logs**: View live container output streaming via websockets directly to the browser console.
- **Speed Diagnostics**: Benchmark tokens-per-second throughput across parallel request streams.

## Prerequisites

- **Docker & Docker Compose**: The manager runs as a service and communicates with Docker daemon via `/var/run/docker.sock`.
- **NVIDIA GPU**: Required for running vLLM containers.
- **Python 3.11+**: If running the backend locally outside of Docker.

## Project Structure

- `server.py`: FastAPI server handling endpoints, background tasks, docker client events, config serialization, and websockets.
- `Dockerfile`: Production-ready image configuration.
- `requirements.txt`: Python package requirements.
- `static/`: Frontend visual assets, scripts, and CSS.
- `templates/`: HTML structures and layouts.

## Docker Setup

Mount the Docker socket and target directory paths inside your docker-compose service configuration:

```yaml
services:
  vllm-ui:
    build:
      context: ./vllm-ui
    container_name: vllm-ui
    ports:
      - "8888:8888"
    volumes:
      - ./vllm-ui:/app
      - ./models:/models
      - ./vllm-cache:/vllm-cache
      - ./repos:/repos
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - VLLM_API_URL=http://vllm-server:8000
      - VLLM_CONTAINER_NAME=vllm-server
    restart: unless-stopped
```
