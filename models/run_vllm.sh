#!/bin/sh
python3 /models/patch_vllm.py

python3 /models/patch_vllm.py

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pin to GPU1 only (GPU0 is LMStudio's seat — don't touch it)
export CUDA_VISIBLE_DEVICES=1

exec python3 -m vllm.entrypoints.openai.api_server \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype auto \
  --max-model-len 4096 \
  --model nvidia/Qwen3-8B-NVFP4 \
  --served-model-name Qwen3-8B-NVFP4 \
  --tensor-parallel-size 1 \
  --trust-remote-code
