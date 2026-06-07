#!/bin/sh
python3 /models/patch_vllm.py

exec python3 -m vllm.entrypoints.openai.api_server \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 8192
