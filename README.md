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

## Speculative Decoding & VRAM Optimizations

Running speculative decoding on consumer GPUs (e.g., RTX 3080 10GB) requires careful tuning to avoid Out-Of-Memory (OOM) failures or host RAM swapping under Windows WDDM/WSL2.

### Optimal Configuration for Qwen3.5 4B + 0.8B

- **Base Model**: `cyankiwi/Qwen3.5-4B-AWQ-4bit` (loads in 3.28 GiB).
- **Draft Model**: `Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ` (loads from `/models/Qwen3.5-0.8B-AWQ`).

#### Recommended Startup Command (`run_vllm.sh`):

```bash
#!/bin/sh
python3 /models/patch_vllm.py
exec python3 -m vllm.entrypoints.openai.api_server \
  --model cyankiwi/Qwen3.5-4B-AWQ-4bit \
  --speculative-model /models/Qwen3.5-0.8B-AWQ \
  --num-speculative-tokens 5 \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.70 \
  --kv-cache-dtype fp8 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --served-model-name unsloth/Qwen3.5-4B-MTP-GGUF \
  --trust-remote-code
```

#### Optimization Rules:
1. **Reduce GPU Memory Utilization**: Set `--gpu-memory-utilization` to `0.70` (or `0.75`). vLLM allocates the draft model out of the *remaining* headroom. If the base model utilization is set too high (e.g. `0.85`), there will not be enough memory left for the draft model's weights and activation state, causing an immediate OOM.
2. **Limit Max Model Length**: Set `--max-model-len 2048` or `1024`. This reduces the size of the KV cache, freeing up VRAM.
3. **Use FP8 KV Cache**: Active `--kv-cache-dtype fp8` cuts KV cache memory consumption in half.
4. **Force Eager Mode**: `--enforce-eager` prevents CUDA graph memory pre-allocations from consuming critical VRAM headroom.

