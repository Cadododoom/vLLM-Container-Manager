# vLLM Container Manager — GPU-Less Debugging & Testing Plan

## Executive Summary

This document outlines the path to getting the **vLLM Container Manager (UI layer)** running on a Windows/WSL2 host **without any NVIDIA GPUs available**. The goal is to validate and test everything that doesn't require actual GPU inference: Docker orchestration, model download management, config editing, UI rendering, API endpoints, and container lifecycle management.

---

## Phase 1: Host Environment Audit

### 1.1 Current State
| Component | Status | Notes |
|-----------|--------|-------|
| OS | Windows (WSL2) | Docker Desktop 29.5.2 |
| GPU Hardware | None available | No RTX 5060 Ti or any NVIDIA GPU accessible |
| Docker | Running | Docker Desktop with nvidia-container-toolkit installed (but no GPUs) |
| Git | v2.54.0 | Cloneable |
| Python | Not verified in container | Python 3.11-slim base image |

### 1.2 Known Blocking Issues for GPU-Less Testing

| # | Issue | Severity | Fix Required |
|---|-------|----------|--------------|
| B1 | `docker-compose.yml` reserves ALL GPUs (`count: all`) | **BLOCKING** | Change to `count: 0` or remove GPU reservation entirely for testing |
| B2 | `/var/run/docker.sock` mount won't work on Windows natively | **BLOCKING** | Use Docker Desktop's socket path or run in WSL2 distro |
| B3 | `nvidia-smi` calls will fail with no GPUs | **HIGH** | Wrap all GPU queries in try/except, return graceful fallbacks |
| B4 | vLLM server container requires CUDA/GPU to start inference | **EXPECTED** | The vllm-server container WILL fail to load models — this is expected. Test the UI layer only. |
| B5 | `VLLM_USE_V1=0` in docker-compose (deprecated) | MEDIUM | Remove from environment section |

---

## Phase 2: Required Code Changes (GPU-Less Mode)

### 2.1 docker-compose.yml Modifications

```yaml
# CHANGES NEEDED:

# 1. Remove GPU reservation from vllm-server for testing:
#    Change:
#      deploy:
#        reservations:
#          devices:
#            - driver: nvidia
#              count: all
#              capabilities: [gpu]
#    To (for testing):
#      deploy:
#        resources:
#          limits:
#            cpus: '4.0'
#    # Remove the reservations block entirely for GPU-less testing

# 2. Remove deprecated VLLM_USE_V1 env var from vllm-server environment section
#    (It's not in current docker-compose.yml, but if added later, remove it)

# 3. For Windows Docker Desktop compatibility:
#    - Use WSL2 distro for running docker-compose
#    - OR use Docker Desktop's built-in compose (docker compose up)
```

### 2.2 server.py Modifications Needed

| Line Range | Issue | Fix |
|------------|-------|-----|
| ~3578-3641 (`/api/system/gpus`) | `nvidia-smi` will fail | Return `{"success": false, "gpus": [], "count": 0}` gracefully — already handled by try/except chain |
| ~1769-1795 (status endpoint VRAM) | `container.exec_run("nvidia-smi ...")` fails | Already wrapped in try/except — returns empty GPUs list |
| ~3120-3158 (`/ws/logs`) | WebSocket to container logs | Will work if container starts, but vllm-server will crash on boot without GPU |

### 2.3 Minimal server.py Patch (GPU-Less Mode)

Add this near the top of `server.py` after imports:

```python
# GPU-LESS MODE FLAG — set True to skip all GPU-dependent operations
GPU_LESS_MODE = os.environ.get("VLLM_GPU_LESS", "1") == "1"
```

Then wrap the nvidia-smi calls in `/api/status` (line ~1769) with:
```python
if not GPU_LESS_MODE and status == "running":
    # existing nvidia-smi code
```

---

## Phase 3: Testing Outlook — What Can Be Tested Without GPUs

### 3.1 Testable Components (Green = Works)

| Component | Endpoint | Status | Notes |
|-----------|----------|--------|-------|
| **FastAPI Backend** | `GET /api/version` | GREEN | Returns version info, no GPU needed |
| **Config Management** | `GET/POST /api/config` | GREEN | Reads/writes `run_vllm.sh`, no GPU needed |
| **Config Backups** | `GET/POST /api/config/backups` | GREEN | File operations only |
| **Sampling Config** | `GET/POST /api/config/sampling` | GREEN | JSON file I/O |
| **Local Models Scan** | `GET /api/models/local` | GREEN | Scans filesystem directories |
| **Model Delete** | `POST /api/models/delete` | GREEN | Filesystem operations |
| **Model Download** | `POST /api/models/download` | GREEN | Uses huggingface-hub, no GPU needed |
| **Download Status** | `GET /api/models/download/status` | GREEN | In-memory state tracking |
| **GitHub Ops** | All `/api/github/*` endpoints | GREEN | Git clone/pull/file operations |
| **System Updates** | `/api/update/*` | GREEN | Self-update logic, no GPU |
| **Web UI (HTML)** | `GET /` | GREEN | Serves static HTML/CSS/JS |
| **WebSocket Logs** | `WS /ws/logs` | PARTIAL | Works if container is running; vllm-server will crash without GPU but connection works |

### 3.2 Non-Testable Components Without GPUs (Red = Won't Work)

| Component | Endpoint | Status | Reason |
|-----------|----------|--------|--------|
| **vLLM Server Inference** | Any `/v1/*` proxy | RED | Requires CUDA/GPU to load models |
| **VRAM Monitoring** | `GET /api/status` (GPU section) | RED | No GPUs = no VRAM data |
| **nvidia-smi Queries** | Inside status endpoint | RED | No GPU hardware available |
| **Context Verification** | `/api/config/verify_context` | RED | Needs live vLLM server to send inference requests |
| **Auto-Tuning** | `/api/config/auto_tune` | RED | Needs actual VRAM metrics from GPUs |
| **Speed Benchmark** | `/api/benchmark/*` | RED | Needs live inference server |
| **Container Restart (full)** | `/api/container/restart` | PARTIAL | Can start container, but vllm-server will crash on boot without GPU |

---

## Phase 4: Step-by-Step Execution Plan

### Step 1: Prepare Docker Compose for GPU-Less Testing

**File**: `docker-compose.yml`

Changes needed:
```yaml
# REMOVE the GPU reservation block from vllm-server:
#   deploy:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]

# ADD environment variable for GPU-less mode:
environment:
  - VLLM_GPU_LESS=1    # <-- Add this to vllm-ui service
```

### Step 2: Build & Start the UI Container Only

```powershell
# In WSL2 or Docker Desktop terminal:
cd "C:\Users\jeffr\Github Pulls\vLLM-Container-Manager"

# Create required directories
New-Item -ItemType Directory -Force -Path ".\models",".\vllm-cache",".\repos" | Out-Null

# Build and start ONLY the UI container (skip vllm-server for now)
docker compose up -d --no-deps vllm-ui
```

### Step 3: Verify UI Container is Running

```powershell
docker ps | Select-String "vllm-ui"
curl http://localhost:8888/api/version
curl http://localhost:8888/
```

Expected output for `/api/version`:
```json
{"version": "...", "status": "ok"}
```

### Step 4: Test All Non-GPU Endpoints

Run through this checklist:

| # | Command | Expected Result |
|---|---------|-----------------|
| 1 | `curl http://localhost:8888/api/version` | JSON with version info |
| 2 | `curl http://localhost:8888/api/status` | Container status = "not_found" (expected, vllm-server not started) |
| 3 | `curl http://localhost:8888/api/config` | Current run_vllm.sh config parsed |
| 4 | `curl http://localhost:8888/api/models/local` | List of local models from /models and /vllm-cache |
| 5 | `curl http://localhost:8888/api/system/gpus` | `{"success": false, "gpus": [], "count": 0}` (expected) |
| 6 | `curl -X POST http://localhost:8888/api/config -d '{"args":{"model":"test"}}' -H "Content-Type: application/json"` | Config updated successfully |

### Step 5: Test Model Download (No GPU Needed)

```powershell
# Start a small model download test
curl -X POST http://localhost:8888/api/models/download `
  -d '{"repo_id":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","local_dir":"/vllm-cache"}' `
  http://localhost:8888/api/models/download

# Check download status
curl http://localhost:8888/api/models/download/status?download_id=<ID_FROM_ABOVE>
```

### Step 6: Test Config Management

```powershell
# Get current config
curl http://localhost:8888/api/config | ConvertFrom-Json | Format-List

# Update model in config
curl -X POST http://localhost:8888/api/config `
  -d '{"args":{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","tensor-parallel-size":"1","max-model-len":"2048"}}' `
  -H "Content-Type: application/json" `
  http://localhost:8888/api/config

# Verify backup was created
curl http://localhost:8888/api/config/backups
```

### Step 7: Test GitHub Operations (Git Clone)

```powershell
# List available repos
curl http://localhost:8888/api/github/repos?repo=Cadododoom/vLLM-Container-Manager

# Get file content
curl http://localhost:8888/api/github/files?path=server.py
curl http://localhost:8888/api/github/file_content?file=server.py
```

### Step 8: Attempt vllm-server Start (Expected to Fail Gracefully)

```powershell
# Try starting the server container — it WILL fail without GPU, but UI should handle it
curl -X POST http://localhost:8888/api/container/start `
  -H "Content-Type: application/json" `
  http://localhost:8888/api/container/start

# Check status — should show container running but vllm_health = offline/failed
curl http://localhost:8888/api/status | ConvertFrom-Json | Format-List

# Watch logs — should see vLLM fail to initialize without GPU
# docker logs -f vllm-server
```

### Step 9: Test WebSocket Logs Connection

Open browser at `http://localhost:8888` and verify:
- UI loads without errors
- Status panel shows "Container not found" or similar (expected)
- Config form is editable
- Model list shows downloaded models from `/vllm-cache`
- Log viewer connects (may show connection error if vllm-server isn't running)

---

## Phase 5: Known Issues & Workarounds

### Issue 1: Docker Socket on Windows
**Problem**: `/var/run/docker.sock` doesn't exist natively on Windows.
**Workaround**: Use Docker Desktop's WSL2 backend. Run docker-compose from within the WSL2 distro:
```bash
# In WSL2 terminal (e.g., Ubuntu):
cd /mnt/c/Users/jeffr/Github\ Pulls/vLLM-Container-Manager
docker compose up -d vllm-ui
```

### Issue 2: vllm-server Container Crashes Without GPU
**Problem**: The `vllm/vllm-openai:latest` image requires CUDA to start.
**Impact**: Any endpoint that depends on a running vLLM server will show "offline".
**Workaround**: This is EXPECTED behavior. Test the UI layer independently — it should gracefully report "server offline" without crashing.

### Issue 3: Path Translation Between Containers
**Problem**: `./vllm-cache` mounts at `/vllm-cache` in vllm-ui but `/root/.cache/huggingface` in vllm-server.
**Impact**: Model paths shown in UI may differ from what vLLM server sees.
**Workaround**: Document the path mapping clearly; test with `GET /api/models/local` to verify scanning works.

### Issue 4: Rate-Limited HuggingFace Downloads
**Problem**: No HF token configured → rate-limited downloads (~10GB/hour).
**Impact**: Large model downloads will be slow or fail.
**Workaround**: Use small models for testing (e.g., TinyLlama-1.1B ~2.2GB).

---

## Phase 6: Success Criteria

### Minimum Viable Test (MVT) — All Must Pass

| # | Check | Expected Result |
|---|-------|-----------------|
| M1 | `docker compose up -d vllm-ui` | Container starts, no errors |
| M2 | `curl http://localhost:8888/api/version` | Returns valid JSON with version |
| M3 | `curl http://localhost:8888/` | HTML page loads (status 200) |
| M4 | `GET /api/config` | Returns parsed run_vllm.sh config |
| M5 | `POST /api/config` | Updates run_vllm.sh successfully |
| M6 | `GET /api/models/local` | Lists models from cache directory |
| M7 | `GET /api/system/gpus` | Returns `{"success": false, ...}` (graceful) |
| M9 | `POST /api/container/start` | Handles missing container gracefully |
| M10 | Browser UI at :8888 | Loads without JS errors, shows "server offline" state |

### Extended Tests (When GPU Becomes Available Later)

| # | Check | Expected Result |
|---|-------|-----------------|
| E1 | `POST /api/container/start` with GPU available | vllm-server starts successfully |
| E2 | Model loads in < 5 minutes (small model) | Server health = "healthy" |
| E3 | `curl http://localhost:8000/health` | Returns 200 OK |
| E4 | Chat completion via UI or curl | Returns valid response |

---

## Appendix A: Quick Reference Commands

```powershell
# === SETUP ===
cd "C:\Users\jeffr\Github Pulls\vLLM-Container-Manager"
New-Item -ItemType Directory -Force -Path ".\models",".\vllm-cache",".\repos" | Out-Null

# === START UI ONLY (GPU-less) ===
docker compose up -d --no-deps vllm-ui

# === VERIFY ===
curl http://localhost:8888/api/version
curl http://localhost:8888/

# === TEST ENDPOINTS ===
curl http://localhost:8888/api/status
curl http://localhost:8888/api/config
curl http://localhost:8888/api/models/local
curl http://localhost:8888/api/system/gpus

# === STOP ===
docker compose down

# === FULL START (requires GPU) ===
docker compose up -d
```

## Appendix B: File Modification Checklist

| File | Change Required | Priority |
|------|----------------|----------|
| `docker-compose.yml` | Remove GPU reservation from vllm-server for testing | HIGH |
| `server.py` | Add `GPU_LESS_MODE` flag, wrap nvidia-smi calls | MEDIUM |
| `models/run_vllm.sh` | No changes needed (runs inside vllm-server) | LOW |
| `templates/index.html` | May need UI tweaks to show "GPU-less mode" banner | LOW |

## Appendix C: Error Recovery Guide

If something goes wrong during testing:

```powershell
# 1. Clean slate
docker compose down -v
Remove-Item -Recurse -Force ".\models\*", ".\vllm-cache\*" -ErrorAction SilentlyContinue

# 2. Recreate directories
New-Item -ItemType Directory -Force -Path ".\models",".\vllm-cache",".\repos" | Out-Null

# 3. Rebuild UI container
docker compose build vllm-ui --no-cache
docker compose up -d vllm-ui

# 4. Check logs
docker logs vllm-ui --tail 50
```
