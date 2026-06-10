import os
import re
import shlex
import time
import json
import pty
import fcntl
import struct
import termios
import select
import asyncio
import logging
import subprocess
import shutil
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import docker
import httpx
import math
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vllm-ui-backend")

app = FastAPI(title="vLLM Manager WebUI")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_last_verified_context()
    load_last_speed_diagnostic()
    t = threading.Thread(target=vllm_container_monitor_loop, daemon=True)
    t.start()

# In-memory stores for background tasks
active_downloads: Dict[str, Dict[str, Any]] = {}
active_git_operations: Dict[str, Dict[str, Any]] = {}
active_benchmarks: Dict[str, Dict[str, Any]] = {}
active_tunings: Dict[str, Dict[str, Any]] = {}
active_verifications: Dict[str, Dict[str, Any]] = {}
active_restarts: Dict[str, Dict[str, Any]] = {}

# GPU Model Loading Layer — supports pause/resume with last-known-good preservation
active_model_loads: Dict[str, Dict[str, Any]] = {}

PAUSE_FLAG_FILE = "/models/.pause_loading"

def _is_persistently_paused() -> bool:
    """Check if model loading is persistently paused (survives container restarts)."""
    return os.path.exists(PAUSE_FLAG_FILE)

def _set_pause(paused: bool) -> None:
    """Set or clear persistent pause flag on disk."""
    try:
        if paused:
            with open(PAUSE_FLAG_FILE, "w") as f:
                f.write("paused")
            logger.info(f"Model loading persistently paused — flag written to {PAUSE_FLAG_FILE}")
        else:
            if os.path.exists(PAUSE_FLAG_FILE):
                os.remove(PAUSE_FLAG_FILE)
                logger.info(f"Model loading un-paused — flag cleared from {PAUSE_FLAG_FILE}")
    except Exception as e:
        logger.error(f"Failed to update pause flag: {e}")

def _stop_vllm_process(container_name: str) -> bool:
    """Stop the vLLM python process inside container to free GPUs immediately."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        res = container.exec_run("pkill -f 'vllm.entrypoints.openai.api_server' || true")
        if res.exit_code == 0:
            logger.info(f"Stopped vLLM process in {container_name} to free GPUs")
            return True
    except Exception as e:
        logger.error(f"Failed to stop vLLM process: {e}")
    return False

def _get_container_pid(container_name: str) -> Optional[str]:
    """Get the main PID of the container's entrypoint process."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        inspect = container.attrs
        pid = inspect.get("State", {}).get("Pid", 0)
        return str(pid) if pid else None
    except Exception:
        return None

# GPU Model Loading Layer — supports pause/resume with last-known-good preservation
active_model_loads: Dict[str, Dict[str, Any]] = {}

# Global variable to cache the parsed KV cache capacity (parsed from logs)
cached_kv_cache_capacity: Optional[int] = None

# Global persistent cache for context verification
LAST_VERIFIED_CONTEXT_PATH = "/models/last_verified_context.json"
last_verified_context_data = None

# Global persistent cache for speed diagnostic
LAST_SPEED_DIAGNOSTIC_PATH = "/models/last_speed_diagnostic.json"
last_speed_diagnostic_result: Optional[Dict[str, Any]] = None
last_verified_container_start: Optional[str] = None
last_seen_container_key: Optional[str] = None

def load_last_verified_context():
    global last_verified_context_data
    if os.path.exists(LAST_VERIFIED_CONTEXT_PATH):
        try:
            with open(LAST_VERIFIED_CONTEXT_PATH, "r") as f:
                last_verified_context_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load last_verified_context.json: {e}")

def load_last_speed_diagnostic():
    global last_speed_diagnostic_result
    if os.path.exists(LAST_SPEED_DIAGNOSTIC_PATH):
        try:
            with open(LAST_SPEED_DIAGNOSTIC_PATH, "r") as f:
                last_speed_diagnostic_result = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load last_speed_diagnostic.json: {e}")

def trigger_auto_context_verification():
    # 1. Prevent concurrent runs
    for v_id, v_state in active_verifications.items():
        if v_state["status"] == "running":
            logger.info("Auto-verification skipped: verification already running.")
            return
            
    for t_id, t_state in active_tunings.items():
        if t_state["status"] == "running":
            logger.info("Auto-verification skipped: tuning running.")
            return
            
    for r_id, b_state in active_benchmarks.items():
        if b_state["status"] == "running":
            logger.info("Auto-verification skipped: benchmark running.")
            return

    # 2. Get container and active model
    try:
        client = docker.from_env()
        container = client.containers.get(VLLM_CONTAINER_NAME)
        
        # Get actual KV cache capacity
        actual_capacity = 0
        try:
            log_bytes = container.logs()
            log_output = log_bytes.decode('utf-8', errors='replace')
            matches = re.findall(r'GPU KV cache size:\s+([\d,]+)\s+tokens', log_output)
            if matches:
                actual_capacity = int(matches[-1].replace(',', ''))
        except Exception as e:
            logger.error(f"Auto-verification: Failed to query log for KV cache size: {e}")

        # Fallback to metrics
        if actual_capacity <= 0:
            try:
                resp = httpx.get(f"{VLLM_API_URL}/metrics", timeout=2.0)
                if resp.status_code == 200:
                    metrics_text = resp.text
                    block_size = 0
                    num_gpu_blocks = 0
                    block_size_match = re.search(r'block_size="(\d+)"', metrics_text)
                    num_gpu_blocks_match = re.search(r'num_gpu_blocks="(\d+)"', metrics_text)
                    if block_size_match:
                        block_size = int(block_size_match.group(1))
                    if num_gpu_blocks_match:
                        num_gpu_blocks = int(num_gpu_blocks_match.group(1))
                    actual_capacity = block_size * num_gpu_blocks
            except Exception as e:
                logger.error(f"Auto-verification: Failed to query metrics: {e}")
                
        if actual_capacity <= 0:
            logger.warning("Auto-verification skipped: could not determine KV cache capacity.")
            return

        model_resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=2.0)
        if model_resp.status_code != 200:
            logger.warning("Auto-verification skipped: server unhealthy.")
            return
            
        models_data = model_resp.json()
        if "data" not in models_data or len(models_data["data"]) == 0:
            logger.warning("Auto-verification skipped: no active model.")
            return
            
        active_model = models_data["data"][0]["id"]
        
        # Get max-model-len
        config = parse_run_vllm_sh(RUN_VLLM_PATH)
        args = config.get("args", {})
        max_model_len = 32768
        try:
            if "max-model-len" in args:
                max_model_len = int(args["max-model-len"])
        except Exception:
            pass

        # We construct the test steps up to actual_capacity
        max_target = actual_capacity
        default_steps = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072]
        test_steps = [s for s in default_steps if s < max_target]
        if not test_steps or test_steps[-1] < max_target:
            test_steps.append(max_target)

        verification_id = "verify_auto_" + str(int(time.time()))
        active_verifications[verification_id] = {
            "status": "idle",
            "progress": 0,
            "current_step": 0,
            "total_steps": len(test_steps),
            "max_stable_length": 0,
            "results": [],
            "logs": f"[Auto] Logical KV Cache size parsed: {actual_capacity} tokens.\n"
                    f"[Auto] Model Context Limit (max-model-len): {max_model_len} tokens.\n"
                    f"[Auto] Testing limit capped at: {max_target} tokens.\n",
            "error": None
        }
        
        t = threading.Thread(
            target=verify_context_task,
            args=(verification_id, active_model, test_steps, max_model_len, actual_capacity),
            daemon=True
        )
        t.start()
        logger.info(f"Auto-verification triggered successfully with ID: {verification_id}")
        
    except Exception as e:
        logger.error(f"Error in trigger_auto_context_verification: {e}")

last_observed_container_state = None
container_unhealthy_since = None
container_exited_consecutive_checks = 0

def trigger_failsafe_autotune():
    # 1. Prevent concurrent runs
    for t_state in active_tunings.values():
        if t_state["status"] == "running":
            logger.info("Failsafe skipped: auto-tune already running.")
            return
            
    tuning_id = "tune_failsafe_" + str(int(time.time()))
    active_tunings[tuning_id] = {
        "status": "idle",
        "progress": 0,
        "current_value": 0.0,
        "stable_value": None,
        "logs": "[Failsafe Triggered] Container remained unhealthy for > 1800 seconds. Initiating VRAM auto-tuning to recover.\n",
        "error": None
    }
    
    t = threading.Thread(
        target=tune_gpu_memory_task,
        args=(tuning_id, 16), # Target 16 concurrency stability
        daemon=True
    )
    t.start()
    logger.info(f"Failsafe auto-tuning triggered successfully with ID: {tuning_id}")

def vllm_container_monitor_loop():
    global last_observed_container_state, last_verified_container_start, last_seen_container_key, container_unhealthy_since, container_exited_consecutive_checks
    logger.info("Starting background vLLM container monitor loop")
    
    # Wait a few seconds for startup to settle
    time.sleep(5)
    
    while True:
        try:
            client = docker.from_env()
            try:
                container = client.containers.get(VLLM_CONTAINER_NAME)
                status = container.status
            except Exception:
                status = "not_found"
                
            # Check vLLM api health and state attributes
            health = "offline"
            started_at = None
            container_id = None
            if status == "running":
                try:
                    container.reload()
                    container_id = container.id
                    started_at = container.attrs.get("State", {}).get("StartedAt")
                    if container_id and started_at:
                        current_container_key = f"{container_id}_{started_at}"
                        if last_seen_container_key != current_container_key:
                            logger.info(f"Container boot/restart detected ({current_container_key}). Resetting unhealthy timer.")
                            container_unhealthy_since = time.time()
                            last_seen_container_key = current_container_key
                except Exception as e:
                    logger.error(f"Failed to reload container state: {e}")
                try:
                    resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
                    if resp.status_code == 200:
                        health = "healthy"
                except Exception:
                    pass
            
            current_state = f"{status}_{health}"
            
            # Failsafe logic: If container is running/restarting but unhealthy for > 1800 seconds
            if status in ("running", "restarting") and health != "healthy":
                container_exited_consecutive_checks = 0
                if container_unhealthy_since is None:
                    container_unhealthy_since = time.time()
                    logger.info("vLLM container is unhealthy/loading. Starting failsafe timer.")
                else:
                    elapsed = time.time() - container_unhealthy_since
                    if elapsed > 1800:
                        logger.warning(f"vLLM container unhealthy for {elapsed:.1f}s. Triggering failsafe recovery.")
                        container_unhealthy_since = None # Reset
                        trigger_failsafe_autotune()
            elif status == "exited" and health != "healthy":
                # Increment consecutive exited checks (crashed container in restart transition)
                container_exited_consecutive_checks += 1
                if container_exited_consecutive_checks >= 3: # 3 consecutive exited checks (15s) means truly stopped by user
                    if container_unhealthy_since is not None:
                        logger.info("vLLM container remained exited/stopped. Resetting failsafe timer.")
                    container_unhealthy_since = None
                    container_exited_consecutive_checks = 0
                else:
                    logger.info(f"vLLM container is exited (check {container_exited_consecutive_checks}/3). Preserving failsafe timer.")
            else:
                if container_unhealthy_since is not None:
                    logger.info("vLLM container became healthy or stopped. Resetting failsafe timer.")
                container_unhealthy_since = None
                container_exited_consecutive_checks = 0
            
            # Detect container start/restart transitions
            if status == "running" and health == "healthy" and container_id and started_at:
                current_container_key = f"{container_id}_{started_at}"
                if last_verified_container_start != current_container_key:
                    logger.info(f"New vLLM container start/restart detected ({current_container_key}). Triggering auto speed check.")
                    last_verified_container_start = current_container_key
                    
                    # Run speed check in background (context scan is user-initiated via UI)
                    try:
                        model_resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=1.0)
                        if model_resp.status_code == 200:
                            models_data = model_resp.json()
                            if "data" in models_data and len(models_data["data"]) > 0:
                                active_model = models_data["data"][0]["id"]
                                def run_bg_speed_check(model_id):
                                    try:
                                        logger.info(f"Running startup speed check for model: {model_id}")
                                        run_speed_diagnostic_internal(model_id)
                                    except Exception as ex:
                                        logger.error(f"Error in background speed check: {ex}")
                                threading.Thread(target=run_bg_speed_check, args=(active_model,), daemon=True).start()
                    except Exception as e:
                        logger.error(f"Failed to query model for startup speed check: {e}")
            
            last_observed_container_state = current_state
            
        except Exception as e:
            logger.error(f"Error in container monitor loop: {e}")
            
        time.sleep(5)



# Constants
VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", "vllm-server")
VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://vllm-server:8000")
RUN_VLLM_PATH = "/models/run_vllm.sh"

# Helper: Parse run_vllm.sh
def parse_run_vllm_sh(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {"preamble": "#!/bin/sh\n", "args": {}}
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read config file: {e}")
        return {"preamble": "#!/bin/sh\n", "args": {}, "error": str(e)}

    lines = content.splitlines()
    preamble_lines = []
    vllm_command_lines = []
    in_vllm_command = False
    
    for line in lines:
        stripped = line.strip()
        if not in_vllm_command:
            if "vllm.entrypoints.openai.api_server" in line:
                in_vllm_command = True
                vllm_command_lines.append(line)
            else:
                preamble_lines.append(line)
        else:
            vllm_command_lines.append(line)
            
    # Combine backslash-ended lines
    combined_cmd = ""
    for line in vllm_command_lines:
        stripped = line.strip()
        if stripped.endswith("\\"):
            combined_cmd += stripped[:-1] + " "
        else:
            combined_cmd += stripped + " "
            
    # Parse combined_cmd using shlex
    try:
        parts = shlex.split(combined_cmd)
    except Exception as e:
        logger.warning(f"shlex parsing failed, falling back to split: {e}")
        parts = combined_cmd.split()
        
    # Find where arguments start
    args_start_idx = -1
    for idx, part in enumerate(parts):
        if "api_server" in part:
            args_start_idx = idx + 1
            break
            
    vllm_args = {}
    if args_start_idx != -1:
        vllm_parts = parts[args_start_idx:]
        i = 0
        while i < len(vllm_parts):
            part = vllm_parts[i]
            if part.startswith("--"):
                key = part[2:]
                # Check if next part is value or another flag
                if i + 1 < len(vllm_parts) and not vllm_parts[i+1].startswith("--"):
                    # Check if value represents an integer, float, or bool
                    val = vllm_parts[i+1]
                    if val.lower() == "true":
                        vllm_args[key] = True
                    elif val.lower() == "false":
                        vllm_args[key] = False
                    elif re.match(r"^\d+$", val):
                        vllm_args[key] = int(val)
                    elif re.match(r"^\d+\.\d+$", val):
                        vllm_args[key] = float(val)
                    else:
                        if key == "speculative-config" and val.startswith("{") and val.endswith("}"):
                            try:
                                import json
                                parts_list = val[1:-1].split(",")
                                normalized_dict = {}
                                for pair in parts_list:
                                    if ":" in pair:
                                        k, v_raw = pair.split(":", 1)
                                        k = k.strip().strip("'\"")
                                        v_raw = v_raw.strip().strip("'\"")
                                        if v_raw.isdigit():
                                            normalized_dict[k] = int(v_raw)
                                        elif v_raw.lower() == "true":
                                            normalized_dict[k] = True
                                        elif v_raw.lower() == "false":
                                            normalized_dict[k] = False
                                        else:
                                            normalized_dict[k] = v_raw
                                val = json.dumps(normalized_dict)
                            except Exception as e:
                                logger.error(f"Failed to normalize speculative-config: {e}")
                        vllm_args[key] = val
                    i += 2
                else:
                    vllm_args[key] = True
                    i += 1
            else:
                i += 1
                
    preamble = "\n".join(preamble_lines) + "\n"
    return {"preamble": preamble, "args": vllm_args}

# Helper: Serialize run_vllm.sh
def serialize_run_vllm_sh(preamble: str, args: Dict[str, Any]) -> str:
    content = preamble
    if not content.endswith("\n"):
        content += "\n"
    content += "python3 /models/patch_vllm.py\n\n"
    content += "exec python3 -m vllm.entrypoints.openai.api_server \\\n"
    
    # Sort keys for consistent serialization
    keys = sorted(list(args.keys()))
    for idx, key in enumerate(keys):
        val = args[key]
        if val is False:
            continue  # Don't output false flags
            
        line = f"  --{key}"
        if val is not True:
            val_str = str(val)
            # Quote values with space or special chars (including JSON braces and quotes)
            if " " in val_str or "*" in val_str or ";" in val_str or "{" in val_str or '"' in val_str:
                line += f" '{val_str}'"
            else:
                line += f" {val_str}"
        
        if idx < len(keys) - 1:
            line += " \\"
        content += line + "\n"
        
    return content

def get_gpu_total_vram() -> int:
    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        if container.status == "running":
            res = container.exec_run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
            if res.exit_code == 0:
                lines = [line.strip() for line in res.output.decode('utf-8').strip().split('\n') if line.strip()]
                if lines:
                    return int(lines[0])
    except Exception as e:
        logger.error(f"Failed to get container VRAM: {e}")
        
    try:
        res = subprocess.run(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        if res.returncode == 0:
            lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
            if lines:
                return int(lines[0])
    except Exception as e:
        logger.error(f"Failed to get local VRAM: {e}")
        
    # Fallback to standard 10GB (10240 MB)
    return 10240

def resolve_model_config_and_size(model_path_or_id: str) -> Dict[str, Any]:
    layers = 32
    kv_heads = 8
    head_dim = 128
    weight_size_gb = 4.0
    model_name = os.path.basename(model_path_or_id)
    model_type = "HF/Transformers"
    config_found = False

    # Check if model is a GGUF file
    if model_path_or_id.endswith(".gguf"):
        model_type = "GGUF"
        model_name = os.path.basename(model_path_or_id)
        local_path = model_path_or_id
        if local_path.startswith("/root/.cache/huggingface/"):
            local_path = local_path.replace("/root/.cache/huggingface/", "/vllm-cache/")
        if not os.path.isabs(local_path):
            local_path = os.path.join("/models", local_path)
        if os.path.exists(local_path):
            try:
                weight_size_gb = os.path.getsize(local_path) / (1024 ** 3)
            except Exception as e:
                logger.error(f"Failed to get GGUF size: {e}")
        return {
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "weight_size_gb": round(weight_size_gb, 3),
            "name": model_name,
            "type": model_type,
            "config_found": False
        }

    search_path = None
    # Translate container cache path if needed for non-GGUF absolute paths
    check_path = model_path_or_id
    if check_path.startswith("/root/.cache/huggingface/"):
        check_path = check_path.replace("/root/.cache/huggingface/", "/vllm-cache/")
    if os.path.isabs(check_path) and os.path.exists(check_path):
        search_path = check_path
    else:
        local_dir = os.path.join("/models", model_path_or_id)
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            search_path = local_dir
            model_type = "Local Directory"
        else:
            repo_folder = f"models--{model_path_or_id.replace('/', '--')}"
            cache_snapshots = None
            for p in ["/vllm-cache", "/vllm-cache/hub"]:
                cand = os.path.join(p, repo_folder, "snapshots")
                if os.path.exists(cand):
                    cache_snapshots = cand
                    break
                    
            if cache_snapshots and os.path.exists(cache_snapshots):
                try:
                    snaps = [s for s in os.scandir(cache_snapshots) if s.is_dir()]
                    if snaps:
                        search_path = snaps[0].path
                        model_type = "HF Cache"
                except Exception as e:
                    logger.error(f"Error checking cache snapshots: {e}")

    if search_path:
        total_size = 0
        for root, _, files in os.walk(search_path):
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
        if total_size > 0:
            weight_size_gb = total_size / (1024 ** 3)

        config_path = os.path.join(search_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                
                layers = cfg.get("num_hidden_layers", cfg.get("num_layers", layers))
                kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", kv_heads))
                
                head_dim = cfg.get("head_dim")
                if not head_dim:
                    hidden_size = cfg.get("hidden_size")
                    num_heads = cfg.get("num_attention_heads")
                    if hidden_size and num_heads:
                        head_dim = hidden_size // num_heads
                    else:
                        head_dim = 128
                        
                if "quantization_config" in cfg:
                    q_method = cfg["quantization_config"].get("quant_method")
                    if q_method:
                        model_type += f" ({q_method.upper()})"
                
                config_found = True
            except Exception as e:
                logger.error(f"Failed to read/parse config.json: {e}")

    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "weight_size_gb": round(weight_size_gb, 3),
        "name": model_name,
        "type": model_type,
        "config_found": config_found
    }

def verify_context_task(verification_id: str, active_model: str, test_steps: List[int], max_model_len: int, actual_capacity: int):
    import concurrent.futures
    state = active_verifications.get(verification_id)
    if not state:
        return
        
    state["status"] = "running"
    state["logs"] += "Starting Real-World Context Capacity Verification\n"
    state["logs"] += f"Active Model: {active_model}\n"
    state["logs"] += f"Steps to test: {test_steps}\n"
    state["logs"] += f"Max Model Length (Single Agent Limit): {max_model_len}\n"
    state["logs"] += f"Logical KV Cache size: {actual_capacity}\n"
    state["logs"] += "--------------------------------------------------\n"
    
    max_stable = 0
    state["total_steps"] = len(test_steps)
    
    # 1. Baseline 1-Agent speed check
    state["logs"] += "Running baseline 1-Agent speed diagnostic...\n"
    baseline_speed = 0.0
    baseline_res = asyncio.run(run_batch(
        concurrency=1,
        model=active_model,
        prompt="Write a 3-sentence summary about artificial intelligence.",
        max_tokens=64
    ))
    if baseline_res.get("success"):
        baseline_speed = baseline_res["throughput"]
        state["logs"] += f"  => Baseline Speed: {baseline_speed:.1f} tokens/s\n"
        if baseline_speed < 15.0:
            state["logs"] += f"  [WARNING] Baseline speed is extremely low ({baseline_speed:.1f} tok/s), indicating host RAM spillover is already present at startup!\n"
    else:
        state["logs"] += f"  => Baseline speed check failed: {baseline_res.get('error', 'Unknown error')}\n"

    # If baseline is already spilled over, we fail immediately to prevent lockups and save time
    if baseline_speed > 0 and baseline_speed < 15.0:
        state["logs"] += "  => FAILED: System is already running in a host RAM swap/spillover state. Verification aborted.\n"
        state["status"] = "failed"
        state["error"] = "System is already running in a host RAM swap/spillover state on baseline check."
        state["progress"] = 100
        # Write to last_verified_context.json
        try:
            result_data = {
                "max_stable_length": 0,
                "timestamp": int(time.time()),
                "model": active_model,
                "status": "failed",
                "kv_cache_capacity": actual_capacity,
                "max_model_len": max_model_len,
                "results": [],
                "logs": state["logs"]
            }
            with open(LAST_VERIFIED_CONTEXT_PATH, "w") as f:
                json.dump(result_data, f, indent=2)
            load_last_verified_context()
        except Exception as e:
            logger.error(f"Failed to write persistent verified context file: {e}")
        return

    for idx, length in enumerate(test_steps):
        state["current_step"] = idx + 1
        state["progress"] = int((idx / len(test_steps)) * 90)
        
        # Determine prompt size and max_tokens for each concurrent request
        chunks = []
        temp_len = length
        while temp_len > 0:
            chunk = min(temp_len, max_model_len)
            chunks.append(chunk)
            temp_len -= chunk

        num_requests = len(chunks)
        state["logs"] += f"Step {idx+1}/{len(test_steps)}: Testing context length {length} tokens ({num_requests} request(s): {chunks} tokens)...\n"

        # Vary the prompt tokens per request to prevent prefix caching from sharing memory between concurrent requests
        def send_req(req_idx):
            # Stagger requests to prevent concurrent prefill scheduler deadlocks in vLLM
            if req_idx > 0:
                time.sleep(req_idx * 2.0)

            chunk_len = chunks[req_idx]
            max_tokens = 16
            prompt_len = chunk_len - max_tokens
            if prompt_len <= 0:
                prompt_len = chunk_len
                max_tokens = 0

            request_prompt = [10 + req_idx] + [1] * (prompt_len - 1)
            payload = {
                "model": active_model,
                "prompt": request_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True
            }
            with httpx.Client() as client:
                resp = client.post(
                    f"{VLLM_API_URL}/v1/completions",
                    json=payload,
                    timeout=120.0
                )
            return resp

        start_time = time.time()
        success = True
        err_msg = ""
        responses = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = {executor.submit(send_req, i): i for i in range(num_requests)}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    responses.append(res)
                    if res.status_code != 200:
                        success = False
                        err_msg = f"HTTP {res.status_code}: {res.text}"
        except httpx.TimeoutException:
            success = False
            err_msg = "Request timed out (potential GPU kernel deadlock/OOM)."
        except Exception as e:
            success = False
            err_msg = f"Error: {str(e) or type(e).__name__}"
            
        elapsed = time.time() - start_time
        
        if success:
            # 2. Run post-step 1-Agent speed diagnostic to verify no spillover occurred
            state["logs"] += "  Running post-step 1-Agent speed check...\n"
            step_speed_res = asyncio.run(run_batch(
                concurrency=1,
                model=active_model,
                prompt="Write a 3-sentence summary about artificial intelligence.",
                max_tokens=64
            ))
            
            step_speed = 0.0
            if step_speed_res.get("success"):
                step_speed = step_speed_res["throughput"]
                state["logs"] += f"  => Post-step 1-Agent Speed: {step_speed:.1f} tokens/s\n"
                
                # Check for RAM spillover / swap
                if step_speed < 15.0:
                    success = False
                    err_msg = f"Post-step speed of {step_speed:.1f} tok/s fell below 15.0 tok/s threshold (Host RAM swap detected)."
                elif baseline_speed > 0:
                    drop_pct = (1 - step_speed / baseline_speed) * 100
                    if drop_pct > 33.0:
                        success = False
                        err_msg = f"Post-step speed dropped by {drop_pct:.1f}% ({step_speed:.1f} tok/s vs baseline {baseline_speed:.1f} tok/s), indicating host RAM spillover."
            else:
                success = False
                err_msg = f"Post-step speed diagnostic failed: {step_speed_res.get('error', 'Unknown error')}"
            
            if success:
                max_stable = length
                state["max_stable_length"] = max_stable
                
                total_processed = length
                tokens_per_sec = total_processed / elapsed if elapsed > 0 else 0
                
                step_result = {
                    "length": length,
                    "status": "success",
                    "duration": round(elapsed, 2),
                    "tokens_per_second": round(tokens_per_sec, 1),
                    "agent_speed": round(step_speed, 1),
                    "error": None
                }
                state["results"].append(step_result)
                state["logs"] += f"  => SUCCESS: processed {total_processed} tokens across {num_requests} request(s) in {elapsed:.2f}s (concurrency throughput: {tokens_per_sec:.1f} tok/s, 1-agent speed: {step_speed:.1f} tok/s)\n"
            
        if not success:
            state["logs"] += f"  => FAILED: {err_msg}\n"
            if "timed out" in err_msg or "deadlock" in err_msg:
                state["logs"] += "  [WARNING] The vLLM server may be deadlocked or overloaded. Restarting the vLLM container is highly recommended.\n"
            step_result = {
                "length": length,
                "status": "failed",
                "duration": round(elapsed, 2),
                "tokens_per_second": 0.0,
                "agent_speed": 0.0,
                "error": err_msg
            }
            state["results"].append(step_result)
            break
            
    state["progress"] = 100
    if max_stable > 0:
        state["status"] = "completed"
        state["logs"] += f"\nVerification finished! Max stable context length: {max_stable} tokens.\n"
    else:
        state["status"] = "failed"
        state["error"] = "Failed to verify even the minimum context length."
        state["logs"] += f"\nVerification failed! Could not verify minimum context length.\n"

    # Persistent write to last_verified_context.json
    try:
        result_data = {
            "max_stable_length": max_stable,
            "timestamp": int(time.time()),
            "model": active_model,
            "status": state["status"],
            "kv_cache_capacity": actual_capacity,
            "max_model_len": max_model_len,
            "results": state["results"],
            "logs": state["logs"]
        }
        with open(LAST_VERIFIED_CONTEXT_PATH, "w") as f:
            json.dump(result_data, f, indent=2)
        load_last_verified_context()
    except Exception as e:
        logger.error(f"Failed to write persistent verified context file: {e}")

def tune_gpu_memory_task(tuning_id: str, target_concurrency: int):
    state = active_tunings.get(tuning_id)
    if not state:
        return
        
    state["status"] = "running"
    state["logs"] += f"Starting GPU Memory Auto-Tuner\n"
    state["logs"] += f"Target concurrency seats to test: {target_concurrency}\n"
    state["logs"] += "--------------------------------------------------\n"
    
    orig_config = parse_run_vllm_sh(RUN_VLLM_PATH)
    if "error" in orig_config:
        state["status"] = "failed"
        state["error"] = orig_config["error"]
        state["logs"] += f"[Error] Failed to parse {RUN_VLLM_PATH}: {orig_config['error']}\n"
        return
        
    preamble = orig_config.get("preamble", "#!/bin/sh\n")
    args = orig_config.get("args", {}).copy()
    model = args.get("model")
    
    if not model:
        state["status"] = "failed"
        state["error"] = "No model selected in run_vllm.sh"
        state["logs"] += "[Error] No model is configured in run_vllm.sh. Please set a model first.\n"
        return
        
    state["logs"] += f"Testing model: {model}\n"
    
    # Binary Search range pre-calculation and safeguards
    stable_value = None
    baseline_speed = 0.0
    original_util = args.get("gpu-memory-utilization")
    
    # Calculate limits based on base and draft model sizes
    vram_total_mb = get_gpu_total_vram()
    vram_total_gb = vram_total_mb / 1024.0
    resolved_base = resolve_model_config_and_size(model)
    
    # Detect speculative decoding configuration
    speculative_mode = "disabled"
    draft_model_path = None
    draft_weight_size_gb = 0.0
    draft_name = "None"
    
    spec_config_str = args.get("speculative-config")
    spec_model_arg = args.get("speculative-model")
    
    if spec_config_str:
        try:
            clean = spec_config_str.strip()
            if clean.startswith("'") and clean.endswith("'"):
                clean = clean[1:-1]
            spec_cfg = json.loads(clean)
            draft_model_path = spec_cfg.get("model")
            if draft_model_path == "[draft]":
                speculative_mode = "mtp_head"
            elif draft_model_path:
                speculative_mode = "draft_model"
        except Exception:
            draft_model_path = spec_config_str
            if draft_model_path == "[draft]":
                speculative_mode = "mtp_head"
            elif draft_model_path:
                speculative_mode = "draft_model"
    elif spec_model_arg:
        draft_model_path = spec_model_arg
        if draft_model_path == "[draft]":
            speculative_mode = "mtp_head"
        elif draft_model_path:
            speculative_mode = "draft_model"
            
    if speculative_mode == "draft_model" and draft_model_path:
        resolved_draft = resolve_model_config_and_size(draft_model_path)
        draft_weight_size_gb = resolved_draft.get("weight_size_gb", 0.0)
        draft_name = resolved_draft.get("name", "None")
        
    tp = 1
    try:
        tp_val = args.get("tensor-parallel-size")
        if tp_val:
            tp = int(tp_val)
    except Exception:
        pass

    base_weight_gb = resolved_base.get("weight_size_gb", 4.0)
    base_weight_gb_scaled = base_weight_gb / tp
    draft_weight_size_gb_scaled = draft_weight_size_gb / tp

    u_max = 0.95
    if speculative_mode == "draft_model" and draft_weight_size_gb > 0:
        u_max = 1.0 - (draft_weight_size_gb_scaled + 0.5) / vram_total_gb
        u_max = round(u_max, 2)
        u_max = min(0.95, max(0.50, u_max))
        state["logs"] += f"[Auto-Tuner] Pre-calculated maximum safe utilization: {u_max:.2f} for draft model '{draft_name}' ({draft_weight_size_gb:.2f} GB total, sharded to {draft_weight_size_gb_scaled:.2f} GB per GPU).\n"
        
    u_min = (base_weight_gb_scaled + 0.8) / vram_total_gb
    u_min = round(u_min, 2)
    
    if speculative_mode == "draft_model" and draft_weight_size_gb > 0 and u_min > u_max:
        state["status"] = "failed"
        state["error"] = f"Infeasible configuration: base model + draft model exceeds GPU VRAM limits."
        state["logs"] += f"\n[Fatal Error] Infeasible configuration for your GPU hardware!\n"
        state["logs"] += f"  Base model weights + overhead require at least {u_min:.2f} utilization ({base_weight_gb_scaled + 0.8:.2f} GB per GPU).\n"
        state["logs"] += f"  Draft model weights + overhead require at least {draft_weight_size_gb_scaled + 0.5:.2f} GB headroom (restricting utilization to {u_max:.2f}).\n"
        state["logs"] += f"  Since Min Util ({u_min:.2f}) > Max Util ({u_max:.2f}), they cannot run together.\n"
        return
        
    client = docker.from_env()
    
    try:
        # 1. Establish baseline first: try values <= u_max to find first bootable configuration
        baseline_candidates = [c for c in [0.70, 0.75, 0.80, 0.85] if c <= u_max]
        if not baseline_candidates:
            baseline_candidates = [u_max]
        baseline_val = None
        
        for candidate in baseline_candidates:
            state["logs"] += f"\nAttempting to establish GPU Memory baseline at {candidate:.2f}...\n"
            state["current_value"] = candidate
            state["progress"] = int(10 + (candidate - 0.70) * 100)
            
            args["gpu-memory-utilization"] = candidate
            try:
                new_content = serialize_run_vllm_sh(preamble, args)
                with open(RUN_VLLM_PATH, "w", newline="\n") as f:
                    f.write(new_content)
                os.chmod(RUN_VLLM_PATH, 0o755)
            except Exception as e:
                state["logs"] += f"Failed writing config for baseline candidate {candidate:.2f}: {e}\n"
                continue
                
            healthy = False
            log_err = "Timeout waiting for health check"
            MAX_BOOT_ATTEMPTS = 2
            
            for attempt in range(1, MAX_BOOT_ATTEMPTS + 1):
                if attempt > 1:
                    state["logs"] += f"  Retrying boot (attempt {attempt}/{MAX_BOOT_ATTEMPTS})...\n"
                    logger.info(f"[AutoTune] baseline={candidate:.2f} retry attempt {attempt}/{MAX_BOOT_ATTEMPTS}")
                
                state["logs"] += "Restarting vLLM container...\n"
                try:
                    container = client.containers.get(VLLM_CONTAINER_NAME)
                    container.restart()
                except Exception as e:
                    log_err = f"Docker restart failed: {e}"
                    state["logs"] += f"Docker restart failed: {e}\n"
                    if attempt < MAX_BOOT_ATTEMPTS:
                        time.sleep(5)
                        continue
                    break
                
                state["logs"] += f"Waiting for vLLM server to become healthy (up to 1800s)...\n"
                log_err = "Timeout waiting for health check"
                attempt_healthy = False
                
                for sec in range(1800):
                    try:
                        container.reload()
                        if container.status != "running":
                            log_err = f"Container is in status '{container.status}'"
                            break
                        log_tail = container.logs(tail=30).decode('utf-8', errors='replace')
                        if "RuntimeError: Engine core initialization failed" in log_tail:
                            log_err = "RuntimeError: Engine core initialization failed"
                            break
                        if "ValueError:" in log_tail:
                            val_err_msg = "ValueError detected in logs"
                            for line in log_tail.splitlines():
                                if "ValueError:" in line:
                                    val_err_msg = line.strip()
                            log_err = val_err_msg
                            break
                    except Exception as e:
                        log_err = f"Docker container query failed: {e}"
                        break
                        
                    try:
                        resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
                        if resp.status_code == 200:
                            attempt_healthy = True
                            break
                    except Exception:
                        pass
                    time.sleep(1.0)
                
                if attempt_healthy:
                    healthy = True
                    break
                    
                is_deterministic_fail = any(kw in log_err for kw in [
                    "ValueError:", "RuntimeError:", "Container is in status"
                ])
                try:
                    container = client.containers.get(VLLM_CONTAINER_NAME)
                    tail = container.logs(tail=10).decode('utf-8', errors='replace')
                    state["logs"] += f"  Attempt {attempt}/{MAX_BOOT_ATTEMPTS} failed: {log_err}\nLog tail:\n{tail}\n"
                except Exception as e:
                    state["logs"] += f"  Attempt {attempt}/{MAX_BOOT_ATTEMPTS} failed: {log_err} (no container logs: {e})\n"
                
                if is_deterministic_fail:
                    state["logs"] += f"  Deterministic failure detected at {candidate:.2f} — skipping retries.\n"
                    break
                
                if attempt < MAX_BOOT_ATTEMPTS:
                    state["logs"] += f"  Waiting 10s before retry...\n"
                    time.sleep(10)
                    
            if not healthy:
                state["logs"] += f"Baseline candidate {candidate:.2f} startup failed: {log_err}\n"
                continue
                
            state["logs"] += f"vLLM server is healthy at {candidate:.2f}. Running speed diagnostic...\n"
            model_id = model
            try:
                model_resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=2.0)
                if model_resp.status_code == 200:
                    model_id = model_resp.json()["data"][0]["id"]
            except Exception:
                pass
                
            try:
                speed_res = asyncio.run(run_batch(
                    concurrency=1,
                    model=model_id,
                    prompt="Write a 3-sentence summary about artificial intelligence.",
                    max_tokens=64
                ))
                if speed_res.get("success"):
                    throughput = speed_res["throughput"]
                    baseline_speed = throughput
                    state["logs"] += f"  => Established Baseline Speed at {candidate:.2f}: {baseline_speed:.1f} tokens/s\n"
                    
                    if throughput < 15.0:
                        state["logs"] += f"  => Baseline speed check FAILED at {candidate:.2f}: {throughput:.1f} tokens/s (Host RAM swap/spillover detected!)\n"
                        continue
                else:
                    state["logs"] += f"  => Baseline speed diagnostic FAILED: {speed_res.get('error', 'Request failed')}\n"
                    continue
                    
                # Run concurrency stability check
                state["logs"] += f"Initiating concurrency stability test at {candidate:.2f}...\n"
                batch_res = asyncio.run(run_batch(
                    concurrency=target_concurrency,
                    model=model_id,
                    prompt="Verify system stability and maximum context allocation.",
                    max_tokens=32
                ))
                if batch_res.get("success"):
                    state["logs"] += f"SUCCESS: Completed stability test at {candidate:.2f} with {batch_res['completed']} concurrent requests!\n"
                    stable_value = candidate
                    baseline_val = candidate
                    break
                else:
                    state["logs"] += f"FAILED: Baseline concurrency stability test failed at {candidate:.2f}: {batch_res.get('error')}\n"
                    continue
            except Exception as e:
                state["logs"] += f"FAILED: Baseline stability check exception at {candidate:.2f}: {e}\n"
                continue
                
        if baseline_val is None:
            state["logs"] += "\n[Error] Failed to establish a working GPU Memory baseline at any of the base values (0.70, 0.75, 0.80, 0.85).\n"
            state["status"] = "failed"
            state["error"] = "Failed to establish working memory baseline"
            return
            
        # 2. Binary Search on [baseline_val + 0.01, u_max]
        low = round(baseline_val + 0.01, 2)
        high = u_max
        step_count = 1
        max_steps = 6
        MAX_BOOT_ATTEMPTS = 3
        
        while low <= high:
            mid = round((low + high) / 2, 2)
            step_count += 1
            state["current_value"] = mid
            progress_val = min(90, int(15 + ((step_count - 1) / max_steps) * 75))
            state["progress"] = progress_val
            
            log_msg = f"\n[Step {step_count}] Testing GPU Memory Utilization = {mid:.2f} (Search range: [{low:.2f}, {high:.2f}])...\n"
            state["logs"] += log_msg
            logger.info(log_msg.strip())
            
            args["gpu-memory-utilization"] = mid
            try:
                new_content = serialize_run_vllm_sh(preamble, args)
                with open(RUN_VLLM_PATH, "w", newline="\n") as f:
                    f.write(new_content)
                os.chmod(RUN_VLLM_PATH, 0o755)
            except Exception as e:
                state["logs"] += f"Failed writing config for step {step_count}: {e}\n"
                high = round(mid - 0.01, 2)
                continue
                
            healthy = False
            log_err = "Timeout waiting for health check"
            
            for attempt in range(1, MAX_BOOT_ATTEMPTS + 1):
                if attempt > 1:
                    state["logs"] += f"  Retrying boot (attempt {attempt}/{MAX_BOOT_ATTEMPTS})...\n"
                    logger.info(f"[AutoTune] step={mid:.2f} retry attempt {attempt}/{MAX_BOOT_ATTEMPTS}")
                
                state["logs"] += "Restarting vLLM container...\n"
                try:
                    container = client.containers.get(VLLM_CONTAINER_NAME)
                    container.restart()
                except Exception as e:
                    log_err = f"Docker restart failed: {e}"
                    state["logs"] += f"Docker restart failed: {e}\n"
                    if attempt < MAX_BOOT_ATTEMPTS:
                        time.sleep(5)
                        continue
                    break
                
                state["logs"] += f"Waiting for vLLM server to become healthy (up to 1800s)...\n"
                log_err = "Timeout waiting for health check"
                attempt_healthy = False
                
                for sec in range(1800):
                    try:
                        container.reload()
                        if container.status != "running":
                            log_err = f"Container is in status '{container.status}'"
                            break
                        log_tail = container.logs(tail=30).decode('utf-8', errors='replace')
                        if "RuntimeError: Engine core initialization failed" in log_tail:
                            log_err = "RuntimeError: Engine core initialization failed"
                            break
                        if "ValueError:" in log_tail:
                            val_err_msg = "ValueError detected in logs"
                            for line in log_tail.splitlines():
                                if "ValueError:" in line:
                                    val_err_msg = line.strip()
                            log_err = val_err_msg
                            break
                    except Exception as e:
                        log_err = f"Docker container query failed: {e}"
                        break
                        
                    try:
                        resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
                        if resp.status_code == 200:
                            attempt_healthy = True
                            break
                    except Exception:
                        pass
                    time.sleep(1.0)
                
                if attempt_healthy:
                    healthy = True
                    break
                    
                is_deterministic_fail = any(kw in log_err for kw in [
                    "ValueError:", "RuntimeError:", "Container is in status"
                ])
                try:
                    container = client.containers.get(VLLM_CONTAINER_NAME)
                    tail = container.logs(tail=10).decode('utf-8', errors='replace')
                    state["logs"] += f"  Attempt {attempt}/{MAX_BOOT_ATTEMPTS} failed: {log_err}\nLog tail:\n{tail}\n"
                except Exception as e:
                    state["logs"] += f"  Attempt {attempt}/{MAX_BOOT_ATTEMPTS} failed: {log_err} (no container logs: {e})\n"
                
                if is_deterministic_fail:
                    state["logs"] += "  Deterministic failure detected — skipping retries for this step.\n"
                    break
                
                if attempt < MAX_BOOT_ATTEMPTS:
                    state["logs"] += f"  Waiting 10s before retry...\n"
                    time.sleep(10)
                    
            if not healthy:
                state["logs"] += f"Step {step_count} startup failed: {log_err}\n"
                state["logs"] += f"=> REJECTED utilization = {mid:.2f} due to container startup failure. Adjusting search range.\n"
                high = round(mid - 0.01, 2)
                continue
                
            state["logs"] += "vLLM server is healthy. Running 1-Agent speed diagnostic...\n"
            
            try:
                speed_res = asyncio.run(run_batch(
                    concurrency=1,
                    model=model_id,
                    prompt="Write a 3-sentence summary about artificial intelligence.",
                    max_tokens=64
                ))
                
                speed_ok = False
                if speed_res.get("success"):
                    throughput = speed_res["throughput"]
                    
                    if throughput < 15.0:
                        state["logs"] += f"  => Speed diagnostic FAILED: {throughput:.1f} tokens/s (Fell below hard limit 15.0 tok/s. Host RAM swap/spillover detected!)\n"
                    elif baseline_speed > 0 and throughput < (baseline_speed * 0.67):
                        drop_pct = (1 - throughput / baseline_speed) * 100
                        state["logs"] += f"  => Speed diagnostic FAILED: {throughput:.1f} tokens/s (Dropped by {drop_pct:.1f}% from baseline {baseline_speed:.1f} tok/s. Host RAM swap/spillover detected!)\n"
                    else:
                        speed_ok = True
                        state["logs"] += f"  => Speed diagnostic PASSED: {throughput:.1f} tokens/s (No spillover detected)\n"
                else:
                    state["logs"] += f"  => Speed diagnostic FAILED: {speed_res.get('error', 'Request failed')}\n"
                    
                if speed_ok:
                    state["logs"] += "Initiating concurrency stability test...\n"
                    batch_res = asyncio.run(run_batch(
                        concurrency=target_concurrency,
                        model=model_id,
                        prompt="Verify system stability and maximum context allocation.",
                        max_tokens=32
                    ))
                    
                    if batch_res.get("success"):
                        state["logs"] += f"SUCCESS: Completed stability test at {mid:.2f} with {batch_res['completed']} concurrent requests!\n"
                        stable_value = mid
                        if throughput > baseline_speed:
                            baseline_speed = throughput
                            state["logs"] += f"  => Updated Baseline Speed: {baseline_speed:.1f} tokens/s\n"
                        low = round(mid + 0.01, 2)
                    else:
                        state["logs"] += f"FAILED: Concurrency stability test failed: {batch_res.get('error')}.\n"
                        state["logs"] += f"=> REJECTED utilization = {mid:.2f} due to stability test failure. Adjusting search range.\n"
                        high = round(mid - 0.01, 2)
                else:
                    state["logs"] += f"FAILED: Memory utilization rejected due to RAM spillover/slow throughput.\n"
                    state["logs"] += f"=> REJECTED utilization = {mid:.2f} due to speed degradation. Adjusting search range.\n"
                    high = round(mid - 0.01, 2)
                    
            except Exception as e:
                state["logs"] += f"FAILED: Stability check exception: {e}.\n"
                state["logs"] += f"=> REJECTED utilization = {mid:.2f} due to exception. Adjusting search range.\n"
                high = round(mid - 0.01, 2)
                
        if stable_value is not None:
            state["logs"] += f"\n*** FOUND STABLE LIMIT: {stable_value:.2f} ***\n"
            state["logs"] += "Saving optimal configuration and performing final restart...\n"
            args["gpu-memory-utilization"] = stable_value
            new_content = serialize_run_vllm_sh(preamble, args)
            with open(RUN_VLLM_PATH, "w", newline="\n") as f:
                f.write(new_content)
            os.chmod(RUN_VLLM_PATH, 0o755)
            
            try:
                container = client.containers.get(VLLM_CONTAINER_NAME)
                container.restart()
            except Exception:
                pass
                
            state["stable_value"] = stable_value
            state["status"] = "completed"
            state["progress"] = 100
            state["logs"] += "Auto-Tuning complete and container restarted successfully!\n"
        else:
            state["logs"] += "\n[Error] Failed to find any stable memory utilization limit.\n"
            state["logs"] += f"Restoring original value of {original_util}...\n"
            if original_util is not None:
                args["gpu-memory-utilization"] = original_util
            else:
                args.pop("gpu-memory-utilization", None)
                
            new_content = serialize_run_vllm_sh(preamble, args)
            with open(RUN_VLLM_PATH, "w", newline="\n") as f:
                f.write(new_content)
            os.chmod(RUN_VLLM_PATH, 0o755)
            
            try:
                container = client.containers.get(VLLM_CONTAINER_NAME)
                container.restart()
            except Exception:
                pass
                
            state["status"] = "failed"
            state["error"] = "No stable utilization value found"
            state["progress"] = 100
            
    except Exception as e:
        logger.error(f"Auto-tune error: {e}")
        state["status"] = "failed"
        state["error"] = str(e)
        state["logs"] += f"\n[Fatal Error] Tuner exception: {e}\n"
        if original_util is not None:
            args["gpu-memory-utilization"] = original_util
            try:
                new_content = serialize_run_vllm_sh(preamble, args)
                with open(RUN_VLLM_PATH, "w", newline="\n") as f:
                    f.write(new_content)
                container = client.containers.get(VLLM_CONTAINER_NAME)
                container.restart()
            except Exception:
                pass

# API: Resolve Model Configuration and Size
@app.get("/api/models/resolve")
def resolve_model_size(model_path: str):
    return resolve_model_config_and_size(model_path)

# API: Context Capacity GET
@app.get("/api/config/context_capacity")
def get_context_capacity():
    config = parse_run_vllm_sh(RUN_VLLM_PATH)
    args = config.get("args", {})
    model_path = args.get("model")
    gpu_util = args.get("gpu-memory-utilization", 0.88)
    vram_total_mb = get_gpu_total_vram()
    
    # 1. Detect speculative decoding configuration
    speculative_mode = "disabled"
    draft_model_path = None
    draft_weight_size_gb = 0.0
    draft_layers = 0
    draft_kv_heads = 0
    draft_head_dim = 0
    draft_type = "None"
    draft_name = "None"
    draft_config_found = False

    spec_config_str = args.get("speculative-config")
    spec_model_arg = args.get("speculative-model")
    
    if spec_config_str:
        try:
            clean = spec_config_str.strip()
            if clean.startswith("'") and clean.endswith("'"):
                clean = clean[1:-1]
            spec_cfg = json.loads(clean)
            draft_model_path = spec_cfg.get("model")
            if draft_model_path == "[draft]":
                speculative_mode = "mtp_head"
            elif draft_model_path:
                speculative_mode = "draft_model"
        except Exception:
            draft_model_path = spec_config_str
            if draft_model_path == "[draft]":
                speculative_mode = "mtp_head"
            elif draft_model_path:
                speculative_mode = "draft_model"
    elif spec_model_arg:
        draft_model_path = spec_model_arg
        if draft_model_path == "[draft]":
            speculative_mode = "mtp_head"
        elif draft_model_path:
            speculative_mode = "draft_model"

    if speculative_mode == "draft_model" and draft_model_path:
        resolved_draft = resolve_model_config_and_size(draft_model_path)
        draft_weight_size_gb = resolved_draft.get("weight_size_gb", 0.0)
        draft_layers = resolved_draft.get("layers", 0)
        draft_kv_heads = resolved_draft.get("kv_heads", 0)
        draft_head_dim = resolved_draft.get("head_dim", 0)
        draft_type = resolved_draft.get("type", "None")
        draft_name = resolved_draft.get("name", "None")
        draft_config_found = resolved_draft.get("config_found", False)
        
    if not model_path:
        return {
            "model_path": None,
            "gpu_memory_utilization": gpu_util,
            "vram_total_mb": vram_total_mb,
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "weight_size_gb": 4.0,
            "type": "Unknown",
            "name": "None",
            "config_found": False,
            "speculative_mode": speculative_mode,
            "draft_model_path": draft_model_path,
            "draft_weight_size_gb": draft_weight_size_gb,
            "draft_layers": draft_layers,
            "draft_kv_heads": draft_kv_heads,
            "draft_head_dim": draft_head_dim,
            "draft_type": draft_type,
            "draft_name": draft_name,
            "draft_config_found": draft_config_found
        }
        
    resolved = resolve_model_config_and_size(model_path)
    return {
        "model_path": model_path,
        "gpu_memory_utilization": gpu_util,
        "vram_total_mb": vram_total_mb,
        "layers": resolved.get("layers"),
        "kv_heads": resolved.get("kv_heads"),
        "head_dim": resolved.get("head_dim"),
        "weight_size_gb": resolved.get("weight_size_gb"),
        "type": resolved.get("type"),
        "name": resolved.get("name"),
        "config_found": resolved.get("config_found"),
        "speculative_mode": speculative_mode,
        "draft_model_path": draft_model_path,
        "draft_weight_size_gb": draft_weight_size_gb,
        "draft_layers": draft_layers,
        "draft_kv_heads": draft_kv_heads,
        "draft_head_dim": draft_head_dim,
        "draft_type": draft_type,
        "draft_name": draft_name,
        "draft_config_found": draft_config_found
    }

# API: Verify Context Capacity
@app.post("/api/config/verify_context")
def verify_context_capacity(background_tasks: BackgroundTasks):
    # 1. Check if server is running
    try:
        client = docker.from_env()
        container = client.containers.get(VLLM_CONTAINER_NAME)
        container_status = container.status
        if container_status != "running":
            raise HTTPException(status_code=503, detail="vLLM container is not running.")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query container: {str(e)}")

    # 2. Prevent concurrent runs
    for v_id, v_state in active_verifications.items():
        if v_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A context verification session is already running.")
            
    for t_id, t_state in active_tunings.items():
        if t_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot verify context while an auto-tuning session is running.")
            
    for r_id, b_state in active_benchmarks.items():
        if b_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot verify context while a benchmark is running.")

    # 3. Try to get physical capacity from cached variable or container logs
    global cached_kv_cache_capacity
    actual_capacity = 0
    if cached_kv_cache_capacity is not None:
        actual_capacity = cached_kv_cache_capacity
    else:
        try:
            log_bytes = container.logs()
            log_output = log_bytes.decode('utf-8', errors='replace')
            matches = re.findall(r'GPU KV cache size:\s+([\d,]+)\s+tokens', log_output)
            if matches:
                cached_kv_cache_capacity = int(matches[-1].replace(',', ''))
                actual_capacity = cached_kv_cache_capacity
        except Exception as e:
            logger.error(f"Failed to query log for KV cache size: {e}")

    # Fallback to metrics if not parsed from logs
    if actual_capacity <= 0:
        try:
            resp = httpx.get(f"{VLLM_API_URL}/metrics", timeout=2.0)
            if resp.status_code == 200:
                metrics_text = resp.text
                block_size = 0
                num_gpu_blocks = 0
                block_size_match = re.search(r'block_size="(\d+)"', metrics_text)
                num_gpu_blocks_match = re.search(r'num_gpu_blocks="(\d+)"', metrics_text)
                if block_size_match:
                    block_size = int(block_size_match.group(1))
                if num_gpu_blocks_match:
                    num_gpu_blocks = int(num_gpu_blocks_match.group(1))
                actual_capacity = block_size * num_gpu_blocks
        except Exception as e:
            logger.error(f"Failed to query vLLM metrics for capacity: {e}")
            
    if actual_capacity <= 0:
        raise HTTPException(status_code=400, detail="vLLM has not allocated KV cache blocks yet or server is loading.")

    # 4. Query active model
    active_model = None
    try:
        model_resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=2.0)
        if model_resp.status_code == 200:
            models_data = model_resp.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                active_model = models_data["data"][0]["id"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query active model: {str(e)}")
        
    if not active_model:
        raise HTTPException(status_code=400, detail="No active model found running on the server.")

    # 5. Get max-model-len to avoid exceeding model bounds
    config = parse_run_vllm_sh(RUN_VLLM_PATH)
    args = config.get("args", {})
    max_model_len = 32768
    try:
        if "max-model-len" in args:
            max_model_len = int(args["max-model-len"])
    except Exception:
        pass

    max_target = actual_capacity
    
    # We define default progression steps up to actual_capacity:
    default_steps = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072]
    test_steps = [s for s in default_steps if s < max_target]
    if not test_steps or test_steps[-1] < max_target:
        test_steps.append(max_target)

    # 6. Initialize verification tracking dict
    verification_id = "verify_" + str(int(time.time()))
    active_verifications[verification_id] = {
        "status": "idle",
        "progress": 0,
        "current_step": 0,
        "total_steps": len(test_steps),
        "max_stable_length": 0,
        "results": [],
        "logs": f"Logical KV Cache size parsed: {actual_capacity} tokens.\n"
                f"Model Context Limit (max-model-len): {max_model_len} tokens.\n"
                f"Testing limit capped at: {max_target} tokens.\n",
        "error": None
    }
    
    background_tasks.add_task(
        verify_context_task,
        verification_id,
        active_model,
        test_steps,
        max_model_len,
        actual_capacity
    )
    
    return {"message": "Context verification started in background", "verification_id": verification_id}

@app.get("/api/config/verify_context/status")
def get_verify_context_status(verification_id: Optional[str] = None):
    if not verification_id:
        if not active_verifications:
            return {"status": "idle", "progress": 0, "max_stable_length": 0, "logs": "", "results": []}
        newest_id = sorted(active_verifications.keys())[-1]
        return {"verification_id": newest_id, **active_verifications[newest_id]}
        
    if verification_id not in active_verifications:
        raise HTTPException(status_code=404, detail="Verification ID not found")
        
    return active_verifications[verification_id]

# API: Start Auto-Tuner
@app.post("/api/config/auto_tune")
def start_auto_tune(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    target_concurrency = payload.get("target_concurrency", 16)
    
    for t_id, t_state in active_tunings.items():
        if t_state["status"] == "running":
            raise HTTPException(status_code=400, detail="An auto-tuning session is already running.")
            
    for r_id, b_state in active_benchmarks.items():
        if b_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot run auto-tune while a benchmark is running.")
            
    tuning_id = "tune_" + str(int(time.time()))
    active_tunings[tuning_id] = {
        "status": "idle",
        "progress": 0,
        "current_value": 0.0,
        "stable_value": None,
        "logs": "",
        "error": None
    }
    
    background_tasks.add_task(
        tune_gpu_memory_task,
        tuning_id,
        target_concurrency
    )
    
    return {"message": "GPU Auto-Tuning started in background", "tuning_id": tuning_id}

# API: Auto-Tuner Status
@app.get("/api/config/auto_tune/status")
def get_auto_tune_status(tuning_id: Optional[str] = None):
    if not tuning_id:
        if not active_tunings:
            return {"status": "idle", "progress": 0, "current_value": 0.0, "logs": ""}
        newest_id = sorted(active_tunings.keys())[-1]
        return {"tuning_id": newest_id, **active_tunings[newest_id]}
        
    if tuning_id not in active_tunings:
        raise HTTPException(status_code=404, detail="Tuning ID not found")
        
    return active_tunings[tuning_id]

# API: Status
@app.get("/api/status")
def get_status():
    global cached_kv_cache_capacity
    
    # Scan active_verifications for any running session
    active_verification = None
    for v_id, v_state in active_verifications.items():
        if v_state.get("status") == "running":
            active_verification = {
                "verification_id": v_id,
                "status": v_state["status"],
                "progress": v_state["progress"],
                "current_step": v_state["current_step"],
                "total_steps": v_state["total_steps"],
                "max_stable_length": v_state["max_stable_length"]
            }
            break

    # Scan active_tunings for any running session (including failsafe ones)
    active_tuning = None
    for t_id, t_state in active_tunings.items():
        if t_state.get("status") == "running":
            active_tuning = {
                "tuning_id": t_id,
                "status": t_state["status"],
                "progress": t_state["progress"],
                "current_value": t_state["current_value"],
                "is_failsafe": t_id.startswith("tune_failsafe_")
            }
            break

    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        status = container.status
        
        # Invalidate cache if container is not running
        if status != "running":
            cached_kv_cache_capacity = None
            
        # Query vLLM health
        health = "offline"
        active_model = "None"
        api_accessible = False
        try:
            # check health
            resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
            if resp.status_code == 200:
                health = "healthy"
                api_accessible = True
            else:
                health = "unhealthy"
        except Exception:
            pass
            
        num_requests_running = 0
        num_requests_waiting = 0
        gpu_cache_usage = 0.0
        num_gpu_blocks = 0
        block_size = 0
        if api_accessible:
            try:
                # Query active model
                model_resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=1.0)
                if model_resp.status_code == 200:
                    models_data = model_resp.json()
                    if "data" in models_data and len(models_data["data"]) > 0:
                        active_model = models_data["data"][0]["id"]
            except Exception:
                pass

            try:
                # Query Prometheus metrics for detailed runtime stats
                metrics_resp = httpx.get(f"{VLLM_API_URL}/metrics", timeout=1.0)
                if metrics_resp.status_code == 200:
                    metrics_text = metrics_resp.text
                    running_match = re.search(r'vllm:num_requests_running\s+(\d+\.?\d*)', metrics_text)
                    waiting_match = re.search(r'vllm:num_requests_waiting\s+(\d+\.?\d*)', metrics_text)
                    cache_match = re.search(r'vllm:gpu_cache_usage_factor\s+(\d+\.?\d*)', metrics_text)
                    
                    if running_match:
                        num_requests_running = int(float(running_match.group(1)))
                    if waiting_match:
                        num_requests_waiting = int(float(waiting_match.group(1)))
                    if cache_match:
                        gpu_cache_usage = float(cache_match.group(1)) * 100.0
                        
                    block_size_match = re.search(r'block_size="(\d+)"', metrics_text)
                    num_gpu_blocks_match = re.search(r'num_gpu_blocks="(\d+)"', metrics_text)
                    if block_size_match:
                        block_size = int(block_size_match.group(1))
                    if num_gpu_blocks_match:
                        num_gpu_blocks = int(num_gpu_blocks_match.group(1))
            except Exception as e:
                logger.error(f"Failed to query metrics: {e}")

        # Parse capacity and memory metrics from logs
        vram_breakdown = None
        loading_status = None
        
        if status == "running":
            try:
                log_output = container.logs().decode('utf-8', errors='replace')
                
                # KV cache capacity
                if health == "healthy" and cached_kv_cache_capacity is None:
                    matches = re.findall(r'GPU KV cache size:\s+([\d,]+)\s+tokens', log_output)
                    if matches:
                        cached_kv_cache_capacity = int(matches[-1].replace(',', ''))
                        
                # VRAM memory breakdown
                if health == "healthy":
                    weights_match = re.search(r'(?:Model weights memory:|Model loading took)\s+([\d.]+)\s*(?:GiB\s+memory|GiB)', log_output)
                    cache_match = re.search(r'(?:Available KV cache memory:|KV Cache memory:)\s+([\d.]+)\s*GiB', log_output)
                    draft_match = re.search(r'(?:Draft model weights memory:|Draft model loading took)\s+([\d.]+)\s*(?:GiB\s+memory|GiB)', log_output)
                    
                    vram_breakdown = {
                        "base_weights_gib": float(weights_match.group(1)) if weights_match else None,
                        "kv_cache_gib": float(cache_match.group(1)) if cache_match else None,
                        "draft_weights_gib": float(draft_match.group(1)) if draft_match else None
                    }
                    
                # Loading monitor status if initializing
                if health != "healthy":
                    log_lines = log_output.splitlines()
                    
                    error_match = None
                    for line in reversed(log_lines[-40:]):
                        if "Error:" in line or "Exception:" in line or "ModuleNotFoundError" in line or "ValueError" in line or "RuntimeError" in line:
                            error_match = line
                            break
                            
                    details = "Initializing engine..."
                    phase = "starting"
                    
                    for line in log_lines[-100:]:
                        if "Downloading" in line or "download" in line.lower() and "%" in line:
                            details = "Downloading model weights..."
                            phase = "downloading"
                        elif "Loading weights" in line or "Loading model weights" in line:
                            details = "Loading base model weights into GPU..."
                            phase = "loading_base"
                        elif "Loading speculative draft" in line or "Loading draft weights" in line:
                            details = "Loading speculative draft model weights..."
                            phase = "loading_draft"
                        elif "Initializing speculative" in line or "speculative_model" in line:
                            details = "Configuring speculative decoding..."
                            phase = "configuring_spec"
                        elif "Initializing KV cache" in line or "allocating KV cache" in line.lower():
                            details = "Allocating GPU KV Cache & profiles..."
                            phase = "kv_cache"
                            
                    if error_match:
                        details = f"Error: {error_match}"
                        phase = "failed"
                        
                    loading_status = {
                        "phase": phase,
                        "details": details,
                        "recent_logs": "\n".join(log_lines[-15:])
                    }
            except Exception as e:
                logger.error(f"Failed to query log specifications: {e}")

        # VRAM and GPU details
        vram_used = 0
        vram_total = 0
        gpu_util = 0
        gpu_name = "NVIDIA GPU"
        gpus = []
        
        if status == "running":
            try:
                # Execute nvidia-smi in vllm-server container
                res = container.exec_run("nvidia-smi --query-gpu=memory.used,memory.total,name,utilization.gpu --format=csv,noheader,nounits")
                if res.exit_code == 0:
                    output = res.output.decode('utf-8').strip()
                    for line in output.splitlines():
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            try:
                                gpus.append({
                                    "vram_used": int(parts[0]),
                                    "vram_total": int(parts[1]),
                                    "name": parts[2],
                                    "utilization": int(parts[3])
                                })
                            except ValueError:
                                pass
                    if gpus:
                        vram_used = sum(g["vram_used"] for g in gpus)
                        vram_total = sum(g["vram_total"] for g in gpus)
                        gpu_util = int(sum(g["utilization"] for g in gpus) / len(gpus))
                        gpu_name = ", ".join(list(set(g["name"] for g in gpus)))
            except Exception as e:
                logger.error(f"Failed to query nvidia-smi: {e}")

        # Try to parse current config to see target model
        config = parse_run_vllm_sh(RUN_VLLM_PATH)
        target_model = config.get("args", {}).get("model", "Unknown")

        return {
            "container_name": VLLM_CONTAINER_NAME,
            "container_status": status,
            "vllm_health": health,
            "active_model": active_model,
            "target_model": target_model,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "gpu_utilization": gpu_util,
            "gpu_name": gpu_name,
            "gpus": gpus,
            "requests_running": num_requests_running,
            "requests_waiting": num_requests_waiting,
            "gpu_cache_usage": gpu_cache_usage,
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": block_size,
            "actual_kv_cache_capacity": cached_kv_cache_capacity,
            "last_verified_context": last_verified_context_data,
            "active_verification": active_verification,
            "active_tuning": active_tuning,
            "last_speed_diagnostic": last_speed_diagnostic_result,
            "loading_status": loading_status,
            "vram_breakdown": vram_breakdown
        }
    except Exception as e:
        return {
            "container_name": VLLM_CONTAINER_NAME,
            "container_status": "not_found",
            "vllm_health": "offline",
            "active_model": "None",
            "target_model": "None",
            "vram_used": 0,
            "vram_total": 0,
            "gpu_utilization": 0,
            "gpu_name": "NVIDIA GPU",
            "gpus": [],
            "requests_running": 0,
            "requests_waiting": 0,
            "gpu_cache_usage": 0.0,
            "num_gpu_blocks": 0,
            "block_size": 0,
            "actual_kv_cache_capacity": None,
            "last_verified_context": last_verified_context_data,
            "active_verification": active_verification,
            "active_tuning": active_tuning,
            "last_speed_diagnostic": last_speed_diagnostic_result,
            "error": str(e)
        }


# API: Local Models
@app.get("/api/models/local")
def get_local_models():
    local_models = []
    
    # 1. Scan /models for GGUFs and directories
    models_dir = "/models"
    if os.path.exists(models_dir):
        for entry in os.scandir(models_dir):
            if entry.is_file() and entry.name.endswith(".gguf"):
                local_models.append({
                    "name": entry.name,
                    "path": f"/models/{entry.name}",
                    "type": "GGUF",
                    "size": entry.stat().st_size,
                    "location": "local"
                })
            elif entry.is_dir():
                # Check for config.json (AWQ or HuggingFace transformers repo format)
                config_path = os.path.join(entry.path, "config.json")
                if os.path.exists(config_path):
                    size = 0
                    for root, _, files in os.walk(entry.path):
                        for f in files:
                            try:
                                size += os.path.getsize(os.path.join(root, f))
                            except Exception:
                                pass
                    local_models.append({
                        "name": entry.name,
                        "path": entry.path,
                        "type": "AWQ/Transformers",
                        "size": size,
                        "location": "local"
                    })

    # 2. Scan /vllm-cache and /vllm-cache/hub for cached HF snapshots
    cache_dirs = ["/vllm-cache", "/vllm-cache/hub"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            for entry in os.scandir(cache_dir):
                if entry.is_dir() and entry.name.startswith("models--"):
                    parts = entry.name.split("--")
                    if len(parts) >= 3:
                        repo_id = "/".join(parts[1:])
                        snapshots_dir = os.path.join(entry.path, "snapshots")
                        if os.path.exists(snapshots_dir):
                            for snap in os.scandir(snapshots_dir):
                                if snap.is_dir():
                                    file_types = set()
                                    gguf_files_found = []
                                    size_total = 0
                                    for root, _, files in os.walk(snap.path):
                                        for f in files:
                                            if f.endswith(".gguf"):
                                                file_types.add("GGUF")
                                                try:
                                                    f_size = os.path.getsize(os.path.join(root, f))
                                                except Exception:
                                                    f_size = 0
                                                gguf_files_found.append((os.path.join(root, f), f_size))
                                            elif f.endswith(".safetensors"):
                                                file_types.add("Safetensors")
                                            try:
                                                # os.path.getsize handles symlinks by resolving to blob file size
                                                size_total += os.path.getsize(os.path.join(root, f))
                                            except Exception:
                                                pass
                                    
                                    if "GGUF" in file_types:
                                        # Yield each individual GGUF file as a model choice
                                        for full_ui_path, f_size in gguf_files_found:
                                            filename = os.path.basename(full_ui_path)
                                            # Translate UI container path /vllm-cache to vllm-server path /root/.cache/huggingface
                                            server_path = full_ui_path
                                            if server_path.startswith("/vllm-cache/"):
                                                server_path = server_path.replace("/vllm-cache/", "/root/.cache/huggingface/")
                                            
                                            # E.g. name: unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
                                            model_display_name = f"{repo_id}/{filename}"
                                            
                                            if not any(m["path"] == server_path for m in local_models):
                                                local_models.append({
                                                    "name": model_display_name,
                                                    "path": server_path,
                                                    "repo_id": repo_id,
                                                    "type": "HF (GGUF)",
                                                    "size": f_size,
                                                    "location": "cache"
                                                })
                                    else:
                                        m_type = "HuggingFace Hub"
                                        if "Safetensors" in file_types:
                                            m_type = "HF (Safetensors)"
                                            
                                        if not any(m["name"] == repo_id for m in local_models):
                                            local_models.append({
                                                "name": repo_id,
                                                "path": repo_id,  # HF models are loaded by repo ID
                                                "repo_id": repo_id,
                                                "type": m_type,
                                                "size": size_total,
                                                "location": "cache"
                                            })
                                    break
                                
    return local_models

@app.post("/api/models/delete")
def delete_model(payload: Dict[str, Any]):
    path = payload.get("path")
    location = payload.get("location")
    
    if not path:
        raise HTTPException(status_code=400, detail="Model path is required")
        
    try:
        import shutil
        if location == "local":
            real_path = os.path.realpath(path)
            if not real_path.startswith(os.path.realpath("/models")):
                raise HTTPException(status_code=403, detail="Can only delete files in /models directory")
                
            if os.path.isfile(real_path):
                os.remove(real_path)
            elif os.path.isdir(real_path):
                shutil.rmtree(real_path)
            else:
                raise HTTPException(status_code=404, detail="Model file or directory not found")
                
        elif location == "cache":
            if path.startswith("/root/.cache/huggingface/") or path.startswith("/vllm-cache/"):
                # Extract the directory name
                path_stripped = path.replace("/root/.cache/huggingface/", "").replace("/vllm-cache/", "")
                parts = [p for p in path_stripped.split("/") if p]
                if parts and parts[0].startswith("models--"):
                    dir_name = parts[0]
                else:
                    raise HTTPException(status_code=400, detail="Invalid cache model path format")
            else:
                if "/" not in path:
                    raise HTTPException(status_code=400, detail="Invalid cache model path")
                parts = path.split("/")
                if len(parts) < 2:
                    raise HTTPException(status_code=400, detail="Invalid cache model path format")
                dir_name = f"models--{parts[0]}--{parts[1]}"
            
            # Check both possible cache locations
            cache_paths = [
                os.path.realpath(os.path.join("/vllm-cache", dir_name)),
                os.path.realpath(os.path.join("/vllm-cache/hub", dir_name))
            ]
            
            deleted = False
            for cache_path in cache_paths:
                # Ensure the path is within the vllm-cache directory to prevent path traversal
                if cache_path.startswith(os.path.realpath("/vllm-cache")):
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                        deleted = True
            
            if not deleted:
                raise HTTPException(status_code=404, detail="Cached model directory not found")
        else:
            raise HTTPException(status_code=400, detail="Invalid location parameter")
            
        return {"status": "ok", "message": f"Successfully deleted model: {path}"}
        
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_model_cache_size(repo_id: str) -> int:
    clean_repo = repo_id.replace("/", "--")
    paths = [
        f"/vllm-cache/models--{clean_repo}",
        f"/vllm-cache/hub/models--{clean_repo}"
    ]
    total_size = 0
    for p in paths:
        if os.path.exists(p):
            try:
                for root, _, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            if os.path.exists(fp):
                                total_size += os.path.getsize(fp)
                        except Exception:
                            pass
            except Exception:
                pass
    return total_size

def cleanup_model_locks(repo_id: str):
    clean_repo = repo_id.replace("/", "--")
    paths = [
        f"/vllm-cache/.locks/models--{clean_repo}",
        f"/vllm-cache/hub/.locks/models--{clean_repo}"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith(".lock"):
                            fp = os.path.join(root, f)
                            try:
                                os.remove(fp)
                                logger.info(f"Removed stale lock file: {fp}")
                            except Exception as e:
                                logger.warning(f"Failed to remove stale lock file {fp}: {e}")
            except Exception as e:
                logger.warning(f"Error cleaning locks directory {p}: {e}")

def parse_hf_url(url_str: str):
    if not url_str:
        return None, None
    url_str = url_str.strip()
    repo_id = url_str
    filename = None
    
    # Check if URL
    if "huggingface.co" in url_str or "hf.co" in url_str:
        # Normalize protocol
        if not url_str.startswith("http://") and not url_str.startswith("https://"):
            url_str = "https://" + url_str
            
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url_str)
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) >= 2:
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                if len(path_parts) > 4 and path_parts[2] in ('blob', 'resolve', 'tree'):
                    filename = "/".join(path_parts[4:])
            
            # Check query params
            query_params = parse_qs(parsed.query)
            if 'show_file_info' in query_params:
                filename = query_params['show_file_info'][0]
        except Exception:
            pass
    elif "/" in url_str:
        # Might be repo/filename format or just repo
        parts = [p for p in url_str.split('/') if p]
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            if len(parts) > 2:
                if len(parts) > 4 and parts[2] in ('blob', 'resolve', 'tree'):
                    filename = "/".join(parts[4:])
                else:
                    filename = "/".join(parts[2:])
                
    if repo_id:
        repo_id = repo_id.split('?')[0].split('#')[0]
    if filename:
        filename = filename.split('?')[0].split('#')[0]
        
    return repo_id, filename

# Background HF Download Task
def download_hf_model_task(
    repo_id: str,
    filename: Optional[str] = None,
    token: Optional[str] = None,
    use_mirror: bool = False,
    enable_transfer: bool = False,
    max_retries: int = 5,
    inactivity_timeout: int = 1200
):
    import queue
    import threading

    active_downloads[repo_id] = {
        "status": "downloading",
        "progress": 0,
        "speed": "N/A",
        "eta": "N/A",
        "logs": f"Initializing download for {repo_id}...\n",
        "error": None
    }

    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_HTTP_TIMEOUT"] = "30"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    if use_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        active_downloads[repo_id]["logs"] += "Mirror enabled: Using https://hf-mirror.com\n"
    if token:
        env["HF_TOKEN"] = token
        # Mask the token for safety in the logs
        masked_token = token[:5] + "..." + token[-4:] if len(token) > 9 else "..."
        active_downloads[repo_id]["logs"] += f"Auth token configured: {masked_token}\n"
    if enable_transfer:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        active_downloads[repo_id]["logs"] += "High-Speed transfer mode enabled (hf-transfer)\n"

    cmd = ["hf", "download", "--cache-dir", "/vllm-cache", repo_id]
    if filename:
        cmd.append(filename)
    if token:
        cmd.extend(["--token", token])

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            active_downloads[repo_id]["logs"] += f"\n--- RETRY ATTEMPT {attempt}/{max_retries} ---\n"
            active_downloads[repo_id]["speed"] = "Retrying"
            active_downloads[repo_id]["eta"] = "N/A"
            logger.info(f"Retrying download for {repo_id} (Attempt {attempt}/{max_retries})")

        try:
            # Note: We do NOT run cleanup_model_locks here to avoid deleting active lock files
            # from parallel or concurrent downloads that other tasks/runs might be waiting for.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            q = queue.Queue()
            def reader():
                try:
                    buffer = []
                    while True:
                        char = process.stdout.read(1)
                        if not char:
                            if buffer:
                                q.put("".join(buffer))
                            break
                        buffer.append(char)
                        if char in ('\n', '\r'):
                            q.put("".join(buffer))
                            buffer = []
                except Exception as ex:
                    q.put(f"[Reader Exception] {ex}\n")
                finally:
                    process.stdout.close()
                    q.put(None) # EOF marker

            t = threading.Thread(target=reader, daemon=True)
            t.start()

            timeout_occurred = False
            last_cache_size = get_model_cache_size(repo_id)
            last_activity_time = time.time()
            poll_interval = 5.0
            
            while True:
                try:
                    # Poll queue with a short timeout to keep UI/logs updated and calculate speed
                    line = q.get(timeout=poll_interval)
                    if line is None:
                        # EOF reached
                        break
                    
                    # We got some output, reset activity timer
                    last_activity_time = time.time()
                    
                    active_downloads[repo_id]["logs"] += line
                    # Limit log buffer size
                    if len(active_downloads[repo_id]["logs"]) > 30000:
                        active_downloads[repo_id]["logs"] = active_downloads[repo_id]["logs"][-30000:]

                    # Parse progress & stats if not in hf-transfer mode
                    if not enable_transfer:
                        # Percentage
                        match = re.search(r"(\d+)%", line)
                        if match:
                            active_downloads[repo_id]["progress"] = int(match.group(1))

                        # tqdm stats
                        tqdm_match = re.search(r"(\d+)%\|.*\|.*\[(.*)<(.*),\s*([^\]]+)\]", line)
                        if tqdm_match:
                            active_downloads[repo_id]["eta"] = tqdm_match.group(3).strip()
                            active_downloads[repo_id]["speed"] = tqdm_match.group(4).strip()
                    else:
                        # In hf-transfer mode, we can parse speed from log output if it's there
                        active_downloads[repo_id]["progress"] = 50  # intermediate progress placeholder
                        active_downloads[repo_id]["speed"] = "High-speed"
                        active_downloads[repo_id]["eta"] = "N/A"

                except queue.Empty:
                    # Queue empty, check if download is making progress via file growth
                    current_cache_size = get_model_cache_size(repo_id)
                    now = time.time()
                    
                    if current_cache_size > last_cache_size:
                        # Download is making progress, reset activity timer
                        last_activity_time = now
                        diff_mb = (current_cache_size - last_cache_size) / (1024 * 1024)
                        speed_mbps = diff_mb / poll_interval
                        
                        # Update progress logs and status for UI
                        active_downloads[repo_id]["speed"] = f"{speed_mbps:.2f} MB/s"
                        active_downloads[repo_id]["eta"] = "Calculating..."
                        
                        active_downloads[repo_id]["logs"] += f"[Progress Check] Cache size grew by {diff_mb:.2f} MB ({speed_mbps:.2f} MB/s)\n"
                        logger.info(f"Download for {repo_id} silent but active: grew by {diff_mb:.2f} MB ({speed_mbps:.2f} MB/s)")
                        last_cache_size = current_cache_size
                    else:
                        # No output and no file growth. Check if inactivity timeout is exceeded
                        elapsed = now - last_activity_time
                        if elapsed >= float(inactivity_timeout):
                            timeout_occurred = True
                            active_downloads[repo_id]["logs"] += f"\n[Warning] Inactivity timeout of {inactivity_timeout}s reached (no logs and no file size growth). Terminating download process...\n"
                            logger.warning(f"Download for {repo_id} timed out after {inactivity_timeout}s of inactivity. Terminating...")
                            
                            process.terminate()
                            time.sleep(2)
                            if process.poll() is None:
                                process.kill()
                            break

            rc = process.wait()

            if timeout_occurred:
                if attempt < max_retries:
                    active_downloads[repo_id]["logs"] += "Waiting 5 seconds before retrying due to timeout...\n"
                    time.sleep(5)
                    continue
                else:
                    active_downloads[repo_id]["status"] = "failed"
                    active_downloads[repo_id]["error"] = f"Download stuck and timed out after {max_retries} attempts."
                    return

            if rc == 0:
                active_downloads[repo_id]["status"] = "completed"
                active_downloads[repo_id]["progress"] = 100
                active_downloads[repo_id]["speed"] = "N/A"
                active_downloads[repo_id]["eta"] = "N/A"
                active_downloads[repo_id]["logs"] += "\n✓ Download completed successfully!\n"
                logger.info(f"Download for {repo_id} completed successfully.")
                return
            else:
                active_downloads[repo_id]["logs"] += f"\n[Error] CLI download exited with code {rc}\n"
                if attempt < max_retries:
                    active_downloads[repo_id]["logs"] += "Waiting 5 seconds before retrying...\n"
                    time.sleep(5)
                    continue
                else:
                    active_downloads[repo_id]["status"] = "failed"
                    active_downloads[repo_id]["error"] = f"CLI download exited with status code {rc} after {max_retries} attempts."
                    return

        except Exception as e:
            active_downloads[repo_id]["logs"] += f"\n[Exception] Error during download attempt: {e}\n"
            logger.error(f"Download attempt failed: {e}")
            if attempt < max_retries:
                time.sleep(5)
                continue
            else:
                active_downloads[repo_id]["status"] = "failed"
                active_downloads[repo_id]["error"] = str(e)
                return

@app.post("/api/models/download")
def download_model(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    repo_id = payload.get("repo_id")
    filename = payload.get("filename")
    token = payload.get("token")
    use_mirror = payload.get("use_mirror", False)
    enable_transfer = payload.get("enable_transfer", False)
    max_retries = int(payload.get("max_retries", 5))
    inactivity_timeout = int(payload.get("inactivity_timeout", 1200))
    
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
        
    # Parse/clean repo_id and filename
    # Let's extract information from both inputs if they are Hugging Face URLs
    r_repo, r_file = parse_hf_url(repo_id)
    
    if filename:
        f_repo, f_file = parse_hf_url(filename)
        
        # Determine final repo_id
        final_repo = r_repo
        if f_repo and ("huggingface.co" in filename or "hf.co" in filename or "/" in filename):
            if not final_repo or "/" not in final_repo or final_repo == repo_id:
                final_repo = f_repo
        
        # Determine final filename
        final_file = f_file if ("huggingface.co" in filename or "hf.co" in filename or "/" in filename) else filename
        if not final_file:
            final_file = r_file
    else:
        final_repo = r_repo
        final_file = r_file
        
    repo_id = final_repo
    filename = final_file

    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
        
    # Clean repo_id
    repo_id = repo_id.strip()
    if not re.match(r"^[\w\-.]+/[\w\-.]+$", repo_id):
        raise HTTPException(status_code=400, detail="Invalid Hugging Face repo format (should be 'username/repo')")
        
    if filename:
        filename = filename.strip()
        
    # Check if download is already in progress
    if repo_id in active_downloads and active_downloads[repo_id]["status"] == "downloading":
        return {"message": "Download already in progress", "repo_id": repo_id}
        
    background_tasks.add_task(
        download_hf_model_task,
        repo_id=repo_id,
        filename=filename,
        token=token,
        use_mirror=use_mirror,
        enable_transfer=enable_transfer,
        max_retries=max_retries,
        inactivity_timeout=inactivity_timeout
    )
    return {"message": "Download started in background", "repo_id": repo_id}

@app.get("/api/models/download/status")
def get_download_status(repo_id: str):
    if repo_id not in active_downloads:
        return {"status": "idle", "progress": 0, "logs": ""}
    return active_downloads[repo_id]

SAMPLING_PARAMS_PATH = "/models/sampling_params.json"

def load_sampling_params() -> Dict[str, Any]:
    default_params = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
    if not os.path.exists(SAMPLING_PARAMS_PATH):
        return default_params
    try:
        with open(SAMPLING_PARAMS_PATH, "r") as f:
            data = json.load(f)
            # Ensure all keys are present
            for k, v in default_params.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        logger.error(f"Failed to read sampling params: {e}")
        return default_params

# Helper: Send a single request for benchmarking
# Helper: Send a single request for benchmarking
async def send_chat_completion_request(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, temperature: Optional[float] = None) -> Optional[tuple]:
    sampling = load_sampling_params()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    
    # Inject sampling configurations
    if temperature is not None:
        payload["temperature"] = float(temperature)
    elif sampling.get("temperature") is not None:
        payload["temperature"] = float(sampling["temperature"])
        
    if sampling.get("top_p") is not None:
        payload["top_p"] = float(sampling["top_p"])
    if sampling.get("top_k") is not None and int(sampling["top_k"]) != -1:
        payload["top_k"] = int(sampling["top_k"])
    if sampling.get("presence_penalty") is not None:
        payload["presence_penalty"] = float(sampling["presence_penalty"])
    if sampling.get("frequency_penalty") is not None:
        payload["frequency_penalty"] = float(sampling["frequency_penalty"])
        
    start = time.perf_counter()
    try:
        url = f"{VLLM_API_URL}/v1/chat/completions"
        response = await client.post(url, json=payload, timeout=90.0)
        end = time.perf_counter()
        if response.status_code == 200:
            res_data = response.json()
            tokens = res_data["usage"]["completion_tokens"]
            duration = end - start
            return duration, tokens
        else:
            logger.error(f"Bench request failed with status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Bench request exception: {e}")
        return None

# Helper: Run a batch of concurrent requests
async def run_batch(concurrency: int, model: str, prompt: str, max_tokens: int, temperature: Optional[float] = None) -> Dict[str, Any]:
    # Set high limits to prevent local socket bottlenecks
    limits = httpx.Limits(max_keepalive_connections=concurrency + 5, max_connections=concurrency + 10)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [send_chat_completion_request(client, model, prompt, max_tokens, temperature) for _ in range(concurrency)]
        start_bench = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_bench = time.perf_counter()
        
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {"success": False, "error": "All requests failed"}
        
    durations, tokens = zip(*valid_results)
    total_tokens = sum(tokens)
    total_time = end_bench - start_bench
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_latency = sum(durations) / len(durations) if len(durations) > 0 else 0
    
    return {
        "success": True,
        "completed": len(valid_results),
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency": avg_latency
    }

# Helper: Query Speculative Decoding Metrics from Prometheus
def get_speculative_decoding_metrics() -> Dict[str, float]:
    try:
        resp = httpx.get(f"{VLLM_API_URL}/metrics", timeout=1.0)
        if resp.status_code == 200:
            text = resp.text
            accepted_match = re.search(r'vllm:spec_decode_num_accepted_tokens(?:_total)?\s+(\d+\.?\d*)', text)
            draft_match = re.search(r'vllm:spec_decode_num_draft_tokens(?:_total)?\s+(\d+\.?\d*)', text)
            
            accepted = float(accepted_match.group(1)) if accepted_match else 0.0
            draft = float(draft_match.group(1)) if draft_match else 0.0
            return {"accepted": accepted, "draft": draft}
    except Exception:
        pass
    return {"accepted": 0.0, "draft": 0.0}

# Task Runner for background thread
def run_benchmark_task(benchmark_id: str, benchmark_type: str, sweep_values: List[Any], model: str, prompt: str, max_tokens: int, fixed_concurrency: int = 2):
    state = active_benchmarks.get(benchmark_id)
    if not state:
        return
        
    state["status"] = "running"
    state["benchmark_type"] = benchmark_type
    state["logs"] += f"Starting Benchmark (Type: {benchmark_type}) for model: {model}\n"
    state["logs"] += f"Target sweep levels: {sweep_values}\n"
    if benchmark_type == "temperature":
        state["logs"] += f"Fixed Concurrency: {fixed_concurrency}\n"
    state["logs"] += f"Max tokens to generate: {max_tokens}\n"
    state["logs"] += "----------------------------------------------\n"
    
    try:
        results = []
        for idx, val in enumerate(sweep_values):
            if state["status"] == "stopping":
                break
                
            if benchmark_type == "temperature":
                current_temp = float(val)
                current_concurrency = fixed_concurrency
                state["current_concurrency"] = f"Temp={current_temp:.2f}"
                log_msg = f"\n--- Testing Temperature: {current_temp:.2f} (Concurrency: {fixed_concurrency}) ---\n"
            else:
                current_temp = None
                current_concurrency = int(val)
                state["current_concurrency"] = f"Conc={current_concurrency}"
                log_msg = f"\n--- Testing Concurrency: {current_concurrency} ---\n"
                
            state["progress"] = int((idx / len(sweep_values)) * 100)
            state["logs"] += log_msg
            logger.info(log_msg.strip())
            
            # Query speculative metrics before batch
            metrics_before = get_speculative_decoding_metrics()
            
            # Execute batch in async loop
            batch_res = asyncio.run(run_batch(current_concurrency, model, prompt, max_tokens, current_temp))
            
            if state["status"] == "stopping":
                break
                
            # Query speculative metrics after batch
            metrics_after = get_speculative_decoding_metrics()
            accepted_diff = metrics_after["accepted"] - metrics_before["accepted"]
            draft_diff = metrics_after["draft"] - metrics_before["draft"]
            
            acceptance_rate = None
            if draft_diff > 0:
                acceptance_rate = round((accepted_diff / draft_diff) * 100.0, 1)
                
            if batch_res.get("success"):
                throughput = batch_res["throughput"]
                avg_latency = batch_res["avg_latency"]
                completed = batch_res["completed"]
                total_time = batch_res["total_time"]
                
                res_entry = {
                    "concurrency": current_concurrency, # Keep concurrency key for chart compatibility
                    "value": val,  # The actual swept variable (temperature or concurrency)
                    "throughput": round(throughput, 2),
                    "latency": round(avg_latency, 2),
                    "completed": completed,
                    "total_time": round(total_time, 2),
                    "acceptance_rate": acceptance_rate
                }
                results.append(res_entry)
                state["results"] = results
                
                success_msg = (
                    f"Completed {completed}/{current_concurrency} requests\n"
                    f"Total Time: {total_time:.2f}s\n"
                    f"Aggregate Throughput: {throughput:.2f} tokens/s\n"
                    f"Avg Latency per Agent: {avg_latency:.2f}s\n"
                )
                if acceptance_rate is not None:
                    success_msg += f"Draft Token Acceptance Rate: {acceptance_rate:.1f}%\n"
                    
                state["logs"] += success_msg
                logger.info(success_msg.strip())
            else:
                err_msg = f"Failed at value {val}: {batch_res.get('error', 'unknown error')}\n"
                state["logs"] += err_msg
                logger.error(err_msg.strip())
                
            # Settle pause
            if idx < len(sweep_values) - 1:
                state["logs"] += "Waiting 3s for cache to settle...\n"
                for _ in range(30):
                    if state["status"] == "stopping":
                        break
                    time.sleep(0.1)
                    
        if state["status"] == "stopping":
            state["status"] = "stopped"
            state["logs"] += "\nBenchmark stopped by user.\n"
        else:
            state["status"] = "completed"
            state["progress"] = 100
            state["logs"] += "\nBenchmark completed successfully!\n"
            
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        state["status"] = "failed"
        state["error"] = str(e)
        state["logs"] += f"\n[Error] Benchmark run failed: {e}\n"

def run_speed_diagnostic_internal(active_model: str) -> Dict[str, Any]:
    global last_speed_diagnostic_result
    prompt = "Write a 3-sentence summary about artificial intelligence."
    max_tokens = 64
    
    batch_res = asyncio.run(run_batch(1, active_model, prompt, max_tokens))
    if not batch_res.get("success"):
        res = {
            "success": False,
            "error": batch_res.get("error", "Request failed"),
            "timestamp": int(time.time()),
            "active_model": active_model
        }
    else:
        throughput = batch_res["throughput"]
        avg_latency = batch_res["avg_latency"]
        spillover = throughput > 0 and throughput < 15.0
        res = {
            "success": True,
            "throughput": round(throughput, 2),
            "latency": round(avg_latency, 2),
            "spillover_detected": spillover,
            "active_model": active_model,
            "timestamp": int(time.time())
        }
        
    last_speed_diagnostic_result = res
    try:
        with open(LAST_SPEED_DIAGNOSTIC_PATH, "w") as f:
            json.dump(res, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write speed diagnostic cache: {e}")
        
    return res

@app.post("/api/benchmark/speed_diagnostic")
def speed_diagnostic():
    try:
        # Check health and get active model
        resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=2.0)
        if resp.status_code != 200:
            return {"success": False, "error": "vLLM server is offline or unhealthy."}
        models_data = resp.json()
        if "data" not in models_data or len(models_data["data"]) == 0:
            return {"success": False, "error": "No active model loaded."}
        active_model = models_data["data"][0]["id"]
        return run_speed_diagnostic_internal(active_model)
    except Exception as e:
        return {"success": False, "error": f"Diagnostic run failed: {str(e)}"}

# API: Start Benchmark
@app.post("/api/benchmark/start")
def start_benchmark(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    benchmark_type = payload.get("benchmark_type", "concurrency")
    concurrency_levels_str = payload.get("concurrency_levels", "1,2,4,8,16,32")
    sweep_values_str = payload.get("sweep_values", concurrency_levels_str)
    fixed_concurrency = int(payload.get("fixed_concurrency", 2))
    model = payload.get("model")
    prompt = payload.get("prompt", "Explain the significance of space exploration in 10 paragraphs.")
    max_tokens = payload.get("max_tokens", 128)
    
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
        
    try:
        if benchmark_type == "temperature":
            sweep_values = [float(x.strip()) for x in sweep_values_str.split(",") if x.strip()]
            sweep_values = sorted([x for x in sweep_values if 0.0 <= x <= 2.5])
        else:
            sweep_values = [int(x.strip()) for x in sweep_values_str.split(",") if x.strip().isdigit()]
            sweep_values = sorted([x for x in sweep_values if 1 <= x <= 256])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sweep values format")
        
    if not sweep_values:
        raise HTTPException(status_code=400, detail="At least one valid sweep value is required")
        
    # Generate unique run ID
    run_id = "run_" + str(int(time.time()))
    
    # Check if a benchmark is already running
    for r_id, b_state in active_benchmarks.items():
        if b_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A benchmark is already running.")
            
    active_benchmarks[run_id] = {
        "status": "idle",
        "progress": 0,
        "current_concurrency": "",
        "logs": "",
        "results": [],
        "error": None
    }
    
    background_tasks.add_task(
        run_benchmark_task,
        run_id,
        benchmark_type,
        sweep_values,
        model,
        prompt,
        max_tokens,
        fixed_concurrency
    )
    
    return {"message": "Benchmark started in background", "run_id": run_id}

# API: Benchmark Status
@app.get("/api/benchmark/status")
def get_benchmark_status(run_id: Optional[str] = None):
    if not run_id:
        if not active_benchmarks:
            return {"status": "idle", "progress": 0, "logs": "", "results": []}
        newest_id = sorted(active_benchmarks.keys())[-1]
        return {"run_id": newest_id, **active_benchmarks[newest_id]}
        
    if run_id not in active_benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark run ID not found")
        
    return active_benchmarks[run_id]

# API: Stop Benchmark
@app.post("/api/benchmark/stop")
def stop_benchmark(payload: Dict[str, Any]):
    run_id = payload.get("run_id")
    if not run_id:
        if not active_benchmarks:
            return {"message": "No active benchmark to stop."}
        run_id = sorted(active_benchmarks.keys())[-1]
        
    if run_id not in active_benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark run ID not found")
        
    if active_benchmarks[run_id]["status"] == "running":
        active_benchmarks[run_id]["status"] = "stopping"
        active_benchmarks[run_id]["logs"] += "\nStopping benchmark... waiting for current step to finish.\n"
        return {"message": "Benchmark stopping request sent."}
        
    return {"message": f"Benchmark is not running (status: {active_benchmarks[run_id]['status']})"}

# API: Config GET

@app.get("/api/config")
def get_config():
    result = parse_run_vllm_sh(RUN_VLLM_PATH)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    custom_jinja_path = "/models/custom_template.jinja"
    custom_jinja = ""
    if os.path.exists(custom_jinja_path):
        try:
            with open(custom_jinja_path, "r", encoding="utf-8") as f:
                custom_jinja = f.read()
        except Exception as e:
            logger.error(f"Failed to read custom template: {e}")
            
    result["custom_jinja"] = custom_jinja
    return result

# API: Config POST
@app.post("/api/config")
def save_config(payload: Dict[str, Any]):
    global cached_kv_cache_capacity
    cached_kv_cache_capacity = None
    args = payload.get("args")
    preamble = payload.get("preamble", "#!/bin/sh\npython3 /models/patch_vllm.py\n")
    custom_jinja = payload.get("custom_jinja")
    
    if args is None:
        raise HTTPException(status_code=400, detail="args payload is required")
        
    # Handle custom jinja template write
    custom_jinja_path = "/models/custom_template.jinja"
    if custom_jinja is not None:
        custom_jinja_stripped = custom_jinja.strip()
        if custom_jinja_stripped:
            try:
                os.makedirs(os.path.dirname(custom_jinja_path), exist_ok=True)
                with open(custom_jinja_path, "w", newline="\n", encoding="utf-8") as f:
                    f.write(custom_jinja_stripped + "\n")
                args["chat-template"] = custom_jinja_path
            except Exception as e:
                logger.error(f"Failed to write custom template file: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save custom chat template: {str(e)}")
        else:
            # If custom_jinja is explicitly sent as empty, clean it up
            if args.get("chat-template") == custom_jinja_path:
                args.pop("chat-template", None)
            try:
                if os.path.exists(custom_jinja_path):
                    os.remove(custom_jinja_path)
            except Exception:
                pass

    if not os.path.exists(RUN_VLLM_PATH):
        # Ensure base directories exist
        os.makedirs(os.path.dirname(RUN_VLLM_PATH), exist_ok=True)
        
    # Create backup before writing
    try:
        if os.path.exists(RUN_VLLM_PATH):
            backup_path = f"{RUN_VLLM_PATH}.bak.{int(time.time())}"
            shutil.copy2(RUN_VLLM_PATH, backup_path)
    except Exception as e:
        logger.error(f"Failed to create config backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")
        
    # Serialize new configuration
    try:
        new_content = serialize_run_vllm_sh(preamble, args)
        with open(RUN_VLLM_PATH, "w", newline="\n") as f:
            f.write(new_content)
        # Ensure executable permissions
        os.chmod(RUN_VLLM_PATH, 0o755)
    except Exception as e:
        logger.error(f"Failed to write config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write config file: {str(e)}")
        
    return {"message": "Config saved successfully. Run vLLM restart to apply.", "file": RUN_VLLM_PATH}

# API: Config backups
@app.get("/api/config/backups")
def list_backups():
    backups = []
    models_dir = "/models"
    if os.path.exists(models_dir):
        for entry in os.scandir(models_dir):
            if entry.is_file() and entry.name.startswith("run_vllm.sh.bak."):
                parts = entry.name.split(".")
                timestamp = int(parts[-1]) if parts[-1].isdigit() else 0
                
                # Parse config details
                parsed = parse_run_vllm_sh(entry.path)
                args = parsed.get("args", {})
                model = args.get("model", "Unknown")
                
                # Parse creator/basename
                creator = "Unknown"
                model_basename = model
                if "/" in model:
                    creator = model.split("/")[0]
                    model_basename = model.split("/")[-1]
                elif model.startswith("/models/"):
                    creator = "Local"
                    model_basename = model.replace("/models/", "")
                
                backups.append({
                    "name": entry.name,
                    "timestamp": timestamp,
                    "formatted_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)) if timestamp else "Unknown",
                    "size": entry.stat().st_size,
                    "model": model,
                    "creator": creator,
                    "model_basename": model_basename,
                    "args": args
                })
    # Sort newest first
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    return backups

@app.post("/api/config/restore")
def restore_backup(payload: Dict[str, Any]):
    global cached_kv_cache_capacity
    cached_kv_cache_capacity = None
    backup_name = payload.get("backup_name")
    if not backup_name:
        raise HTTPException(status_code=400, detail="backup_name is required")
        
    backup_path = os.path.join("/models", backup_name)
    if not os.path.exists(backup_path):
        raise HTTPException(status_code=404, detail="Backup file not found")
        
    try:
        # Create a backup of the current run_vllm.sh first
        if os.path.exists(RUN_VLLM_PATH):
            temp_backup = f"{RUN_VLLM_PATH}.bak.{int(time.time())}"
            shutil.copy2(RUN_VLLM_PATH, temp_backup)
            
        shutil.copy2(backup_path, RUN_VLLM_PATH)
        os.chmod(RUN_VLLM_PATH, 0o755)
    except Exception as e:
        logger.error(f"Failed to restore config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore config: {str(e)}")
        
    return {"message": "Backup restored successfully. Run restart to apply."}

def safe_restart_container_thread(restart_id: str, container_name: str, run_vllm_path: str):
    state = active_restarts[restart_id]
    state["status"] = "running"
    state["logs"] = "Starting health-monitored container restart...\n"
    
    known_good_content = None
    known_good_backup_path = None
    
    # Check if currently healthy
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        health = "offline"
        try:
            resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
            if resp.status_code == 200:
                health = "healthy"
        except Exception:
            pass
            
        if health == "healthy" and os.path.exists(run_vllm_path):
            with open(run_vllm_path, "r", encoding="utf-8") as f:
                known_good_content = f.read()
            state["logs"] += "Current configuration is active and healthy. Stored as recovery point.\n"
    except Exception as e:
        state["logs"] += f"Could not verify current container health: {e}. Checking filesystem for recent backups...\n"

    # Fallback to latest filesystem backup if container is currently offline
    if known_good_content is None:
        models_dir = os.path.dirname(run_vllm_path)
        if os.path.exists(models_dir):
            backups = []
            for entry in os.scandir(models_dir):
                if entry.is_file() and entry.name.startswith("run_vllm.sh.bak."):
                    parts = entry.name.split(".")
                    timestamp = int(parts[-1]) if parts[-1].isdigit() else 0
                    backups.append((entry.path, timestamp))
            if backups:
                backups.sort(key=lambda x: x[1], reverse=True)
                latest_backup = backups[0][0]
                try:
                    with open(latest_backup, "r", encoding="utf-8") as f:
                        known_good_content = f.read()
                    known_good_backup_path = latest_backup
                    state["logs"] += f"Found fallback recovery config backup: {os.path.basename(latest_backup)}\n"
                except Exception as e:
                    state["logs"] += f"Error reading fallback backup: {e}\n"
    
    # 2. Trigger the restart
    restart_time = int(time.time())
    try:
        container = client.containers.get(container_name)
        state["logs"] += f"Initiating Docker container restart for '{container_name}'...\n"
        container.restart()
    except Exception as e:
        state["status"] = "failed"
        state["error"] = f"Docker restart command failed: {e}"
        state["logs"] += f"[Error] Docker restart command failed: {e}\n"
        return

    # 3. Monitor container boot/health (up to 300 seconds) with pause/resume support
    is_new_idle = False
    try:
        if os.path.exists(run_vllm_path):
            with open(run_vllm_path, "r", encoding="utf-8") as f:
                new_content = f.read()
            if "vllm.entrypoints.openai.api_server" not in new_content:
                is_new_idle = True
    except Exception:
        pass

    state["logs"] += "Monitoring startup status...\n"
    healthy = False
    log_err = "Timeout waiting for health check"
    
    # Pause/resume support variables
    paused_at_second = None
    pause_log_snapshot = ""
    persistent_pause_active = _is_persistently_paused()
    
    if persistent_pause_active:
        state["logs"] += "[PAUSE INITIATED] Persistent pause flag detected — pausing model loading immediately...\n"
        # Kill vLLM process immediately on startup if paused
        try:
            _stop_vllm_process(container_name)
            state["logs"] += "[PAUSE INITIATED] vLLM process killed — waiting for resume signal.\n"
        except Exception as e:
            state["logs"] += f"[PAUSE INITIATED] Failed to stop vLLM process: {e}\n"
    
    for sec in range(300):
        time.sleep(1.0)
        
        # Check persistent pause flag — wait for resume signal if paused
        if _is_persistently_paused():
            if paused_at_second is None:
                paused_at_second = sec
                state["logs"] += f"[PAUSED] Model loading paused at second {sec}. Waiting for user to resume...\n"
                
                # Kill vLLM process if running
                try:
                    _stop_vllm_process(container_name)
                    state["logs"] += "[PAUSED] vLLM process stopped — GPUs freed.\n"
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to stop vLLM process: {e}\n"
                
                # Save last known good config on pause
                try:
                    if os.path.exists(run_vllm_path):
                        with open(run_vllm_path, "r", encoding="utf-8") as f:
                            current_config = f.read()
                        if not known_good_content or len(current_config) > len(known_good_content):
                            known_good_content = current_config
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to preserve config: {e}\n"
                
                # Preserve backup on disk
                try:
                    if os.path.exists(run_vllm_path) and known_good_content:
                        backup_path = f"{run_vllm_path}.bak.{int(time.time())}"
                        with open(backup_path, "w", encoding="utf-8") as f:
                            f.write(known_good_content)
                        state["logs"] += f"[PAUSED] Last-known-good config preserved to {os.path.basename(backup_path)}\n"
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to write backup: {e}\n"
                
                # Keep checking for resume — don't break, just skip health checks
                continue
        
        # Check for manual pause request from UI (active_model_loads)
        if restart_id in active_model_loads and active_model_loads[restart_id].get("paused"):
            if paused_at_second is None:
                paused_at_second = sec
                pause_log_snapshot = state["logs"][-2000:] if len(state["logs"]) > 2000 else state["logs"]
                state["logs"] += f"[PAUSED] Model loading paused at second {sec}. Stopping vLLM process to free GPUs...\n"
                
                # Kill the vLLM python process inside container immediately
                try:
                    _stop_vllm_process(container_name)
                    state["logs"] += "[PAUSED] vLLM process stopped — GPUs freed.\n"
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to stop vLLM process: {e}\n"
                
                # Save last known good config on pause
                try:
                    if os.path.exists(run_vllm_path):
                        with open(run_vllm_path, "r", encoding="utf-8") as f:
                            current_config = f.read()
                        if not known_good_content or len(current_config) > len(known_good_content):
                            known_good_content = current_config
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to preserve config: {e}\n"
                
                # Preserve backup on disk
                try:
                    if os.path.exists(run_vllm_path) and known_good_content:
                        backup_path = f"{run_vllm_path}.bak.{int(time.time())}"
                        with open(backup_path, "w", encoding="utf-8") as f:
                            f.write(known_good_content)
                        state["logs"] += f"[PAUSED] Last-known-good config preserved to {os.path.basename(backup_path)}\n"
                except Exception as e:
                    state["logs"] += f"[PAUSED] Failed to write backup: {e}\n"
                
                # Keep checking for resume — don't break, just skip health checks
                continue
        
        # Check for resume request (manual un-pause from UI)
        if restart_id in active_model_loads and not active_model_loads[restart_id].get("paused") and paused_at_second is not None:
            state["logs"] += f"[RESUMED] Resuming model loading from second {paused_at_second}...\n"
            paused_at_second = None
            pause_log_snapshot = ""
        
        try:
            container.reload()
            if container.status != "running":
                log_tail = container.logs(since=restart_time, tail=20).decode('utf-8', errors='replace')
                log_err = f"Container crashed with status '{container.status}'"
                state["logs"] += f"[Warning] Container is in status '{container.status}'. Log tail:\n{log_tail}\n"
                break
                
            log_tail = container.logs(since=restart_time, tail=30).decode('utf-8', errors='replace')
            error_keywords = ["ValueError:", "RuntimeError:", "TypeError:", "NameError:", "AttributeError:", "ImportError:", "ModuleNotFoundError:", "FileNotFoundError:", "PermissionError:", "Traceback (most recent call last):", "Exception:", "vllm: error:", "Error:"]
            found_err = None
            for kw in error_keywords:
                if kw in log_tail:
                    found_err = kw
                    break
            if found_err:
                log_err = "Initialization error detected in logs"
                for line in log_tail.splitlines():
                    if any(ekw in line for ekw in error_keywords):
                        log_err = line.strip()
                        break
                state["logs"] += f"[Warning] Initialization error found in logs: {log_err}\n"
                break
        except Exception as e:
            state["logs"] += f"Failed querying container status: {e}\n"
            
        if is_new_idle:
            if sec >= 5:
                healthy = True
                state["logs"] += "✓ Success: Idle container started successfully.\n"
                break
        else:
            try:
                resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
                if resp.status_code == 200:
                    healthy = True
                    state["logs"] += "✓ Success: Container passed health check and is online!\n"
                    break
            except Exception:
                pass
            
    if healthy:
        state["status"] = "completed"
        return
        
    # 4. Rollback Protocol
    state["logs"] += f"\n[Fatal Update Failure] Health check failed: {log_err}.\n"
    if known_good_content:
        state["logs"] += "[Auto-Recovery] Rolling back configuration file and restarting container with last known-good configuration...\n"
        try:
            with open(run_vllm_path, "w", newline="\n", encoding="utf-8") as f:
                f.write(known_good_content)
            os.chmod(run_vllm_path, 0o755)
            
            container.restart()
            
            rollback_healthy = False
            is_rollback_idle = "vllm.entrypoints.openai.api_server" not in known_good_content
            for sec in range(90):
                time.sleep(1.0)
                try:
                    container.reload()
                    if container.status != "running":
                        break
                    
                    if is_rollback_idle:
                        if sec >= 5:
                            rollback_healthy = True
                            break
                    else:
                        resp = httpx.get(f"{VLLM_API_URL}/health", timeout=1.0)
                        if resp.status_code == 200:
                            rollback_healthy = True
                            break
                except Exception:
                    pass
            if rollback_healthy:
                state["logs"] += "✓ Auto-Recovery Complete: Restored last known-good configuration. Service is online.\n"
                state["status"] = "failed"
                state["error"] = f"New config failed health check ({log_err}). Auto-recovered successfully to previous working state."
            else:
                state["logs"] += "❌ Fatal: Recovery container also failed to start. System requires manual intervention!\n"
                state["status"] = "failed"
                state["error"] = f"New config failed ({log_err}) and recovery config also failed to boot."
        except Exception as rollback_err:
            state["logs"] += f"❌ Critical failure during recovery rollback execution: {rollback_err}\n"
            state["status"] = "failed"
            state["error"] = f"New config failed ({log_err}) and recovery rollback failed: {rollback_err}"
    else:
        state["logs"] += "❌ Failure: No recovery configuration could be found to roll back to.\n"
        state["status"] = "failed"
        state["error"] = f"New config failed health check ({log_err}) and no recovery backup found."

# API: Container control
@app.post("/api/container/restart")
def restart_vllm_container():
    global cached_kv_cache_capacity
    cached_kv_cache_capacity = None
    
    import uuid
    restart_id = str(uuid.uuid4())
    active_restarts[restart_id] = {
        "status": "pending",
        "logs": "Restart requested...\n",
        "error": None
    }
    
    t = threading.Thread(
        target=safe_restart_container_thread,
        args=(restart_id, VLLM_CONTAINER_NAME, RUN_VLLM_PATH),
        daemon=True
    )
    t.start()
    
    return {
        "status": "restarting",
        "restart_id": restart_id,
        "message": f"Container {VLLM_CONTAINER_NAME} health-monitored restart initiated."
    }

@app.get("/api/container/restart/status")
def get_restart_status(restart_id: str):
    state = active_restarts.get(restart_id)
    if not state:
        raise HTTPException(status_code=404, detail="Restart task not found")
    return state

# GPU Pause/Resume — persistent toggle that survives container restarts
@app.get("/api/container/pause/status")
def get_pause_status():
    """Check if model loading is persistently paused (survives across restarts)."""
    return {
        "paused": _is_persistently_paused(),
        "message": "Model loading is paused — settings can be changed freely" if _is_persistently_paused() else "Model loading active"
    }

@app.post("/api/container/pause")
def toggle_pause():
    """Toggle persistent pause state. If already paused, un-pauses it."""
    currently_paused = _is_persistently_paused()
    
    if currently_paused:
        # Un-pause: clear flag and restart container to resume loading
        _set_pause(False)
        
        client = docker.from_env()
        try:
            container = client.containers.get(VLLM_CONTAINER_NAME)
            container.restart()
            return {
                "status": "resuming",
                "paused": False,
                "message": "Model loading resumed — container restarting"
            }
        except docker.errors.NotFound:
            return {
                "status": "resumed_no_container",
                "paused": False,
                "message": "Pause flag cleared. Start container to resume model loading."
            }
        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Pause: set persistent flag and kill vLLM process if running
        _set_pause(True)
        
        client = docker.from_env()
        try:
            container = client.containers.get(VLLM_CONTAINER_NAME)
            stopped = _stop_vllm_process(VLLM_CONTAINER_NAME)
            
            # Preserve last-known-good config
            try:
                if os.path.exists(RUN_VLLM_PATH):
                    with open(RUN_VLLM_PATH, "r", encoding="utf-8") as f:
                        current_config = f.read()
                    backup_path = f"{RUN_VLLM_PATH}.bak.{int(time.time())}"
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(current_config)
            except Exception as e:
                logger.error(f"Failed to preserve config on pause: {e}")
            
            return {
                "status": "paused",
                "paused": True,
                "message": "Model loading paused — GPUs freed. Change settings freely.",
                "gpu_freed": stopped
            }
        except docker.errors.NotFound:
            # Container not running — just set the flag
            return {
                "status": "paused_no_container",
                "paused": True,
                "message": "Pause flag set. Start container to resume model loading."
            }
        except Exception as e:
            logger.error(f"Failed to pause: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/container/resume")
def resume_model_loading():
    """Resume model loading by clearing pause flag and restarting container."""
    _set_pause(False)
    
    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        container.restart()
        return {
            "status": "resuming",
            "paused": False,
            "message": "Model loading resumed — container restarting"
        }
    except docker.errors.NotFound:
        return {
            "status": "resumed_no_container",
            "paused": False,
            "message": "Pause flag cleared. Start container to resume model loading."
        }
    except Exception as e:
        logger.error(f"Failed to resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# GPU MONITOR & SELECTION API ENDPOINTS
# ============================================================

GPU_ASSIGNMENT_FILE = "/models/gpu_assignment.json"
GPU_HISTORY_FILE = "/models/gpu_history.json"

# In-memory ring buffer for 24-hour history (1 data point per minute)
# Max ~1440 points per metric
MAX_HISTORY_POINTS = 1440
gpu_history_buffer: Dict[str, List[Dict]] = {
    "timestamps": [],
    "gpu_vram_used_mb": {},
    "gpu_temp_c": {},
    "gpu_fan_pct": {}
}

def _record_gpu_metrics(gpu_data: List[Dict]) -> None:
    """Record current GPU metrics into the ring buffer."""
    now = time.time()
    
    # Add timestamp (keep only last 1440 points)
    gpu_history_buffer["timestamps"].append(now)
    if len(gpu_history_buffer["timestamps"]) > MAX_HISTORY_POINTS:
        gpu_history_buffer["timestamps"] = gpu_history_buffer["timestamps"][-MAX_HISTORY_POINTS:]
    
    for gpu in gpu_data:
        idx = str(gpu.get("index", 0))
        
        # VRAM used (MB)
        vram_used = gpu.get("vram_used_mb", 0) or 0
        if idx not in gpu_history_buffer["gpu_vram_used_mb"]:
            gpu_history_buffer["gpu_vram_used_mb"][idx] = []
        gpu_history_buffer["gpu_vram_used_mb"][idx].append(vram_used)
        if len(gpu_history_buffer["gpu_vram_used_mb"][idx]) > MAX_HISTORY_POINTS:
            gpu_history_buffer["gpu_vram_used_mb"][idx] = gpu_history_buffer["gpu_vram_used_mb"][idx][-MAX_HISTORY_POINTS:]
        
        # Temperature (°C)
        temp = gpu.get("temp_c", 0) or 0
        if idx not in gpu_history_buffer["gpu_temp_c"]:
            gpu_history_buffer["gpu_temp_c"][idx] = []
        gpu_history_buffer["gpu_temp_c"][idx].append(temp)
        if len(gpu_history_buffer["gpu_temp_c"][idx]) > MAX_HISTORY_POINTS:
            gpu_history_buffer["gpu_temp_c"][idx] = gpu_history_buffer["gpu_temp_c"][idx][-MAX_HISTORY_POINTS:]
        
        # Fan speed (%)
        fan = gpu.get("fan_pct", 0) or 0
        if idx not in gpu_history_buffer["gpu_fan_pct"]:
            gpu_history_buffer["gpu_fan_pct"][idx] = []
        gpu_history_buffer["gpu_fan_pct"][idx].append(fan)
        if len(gpu_history_buffer["gpu_fan_pct"][idx]) > MAX_HISTORY_POINTS:
            gpu_history_buffer["gpu_fan_pct"][idx] = gpu_history_buffer["gpu_fan_pct"][idx][-MAX_HISTORY_POINTS:]

def _persist_gpu_history() -> None:
    """Persist ring buffer to disk file."""
    try:
        payload = {
            "timestamps": gpu_history_buffer["timestamps"][-MAX_HISTORY_POINTS:],
            "gpu_vram_used_mb": {k: v[-MAX_HISTORY_POINTS:] for k, v in gpu_history_buffer["gpu_vram_used_mb"].items()},
            "gpu_temp_c": {k: v[-MAX_HISTORY_POINTS:] for k, v in gpu_history_buffer["gpu_temp_c"].items()},
            "gpu_fan_pct": {k: v[-MAX_HISTORY_POINTS:] for k, v in gpu_history_buffer["gpu_fan_pct"].items()}
        }
        with open(GPU_HISTORY_FILE, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.error(f"Failed to persist GPU history: {e}")

def _load_gpu_history() -> None:
    """Load persisted GPU history from disk on startup."""
    global gpu_history_buffer
    try:
        if os.path.exists(GPU_HISTORY_FILE):
            with open(GPU_HISTORY_FILE, "r") as f:
                data = json.load(f)
            gpu_history_buffer["timestamps"] = data.get("timestamps", [])
            for k, v in data.get("gpu_vram_used_mb", {}).items():
                gpu_history_buffer["gpu_vram_used_mb"][k] = v[-MAX_HISTORY_POINTS:]
            for k, v in data.get("gpu_temp_c", {}).items():
                gpu_history_buffer["gpu_temp_c"][k] = v[-MAX_HISTORY_POINTS:]
            for k, v in data.get("gpu_fan_pct", {}).items():
                gpu_history_buffer["gpu_fan_pct"][k] = v[-MAX_HISTORY_POINTS:]
    except Exception as e:
        logger.error(f"Failed to load GPU history: {e}")

def _get_gpu_assignment() -> Dict[str, Any]:
    """Load current GPU assignment from disk."""
    try:
        if os.path.exists(GPU_ASSIGNMENT_FILE):
            with open(GPU_ASSIGNMENT_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load GPU assignment: {e}")
    return {"tp_slots": [], "pp_slots": []}

def _save_gpu_assignment(data: Dict[str, Any]) -> None:
    """Save GPU assignment to disk."""
    try:
        with open(GPU_ASSIGNMENT_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save GPU assignment: {e}")

def _parse_nvidia_smi() -> List[Dict]:
    """Parse nvidia-smi output into structured GPU metrics with multiple fallback methods."""
    gpus = []
    
    # ============================================================
    # Method 1: Direct subprocess nvidia-smi (Linux) or nvidia-smi.exe (Windows)
    # ============================================================
    try:
        import subprocess
        import sys as _sys
        
        # Find nvidia-smi executable with Windows fallback paths
        nvidia_smi_cmd = "nvidia-smi"
        if _sys.platform == "win32":
            nvidia_smi_cmd = "nvidia-smi.exe"
            # Try common NVIDIA driver installation paths on Windows
            for base_path in [
                r"C:\Program Files\NVIDIA Corporation\NVSMI",
                r"C:\Windows\System32",
            ]:
                try:
                    full_path = os.path.join(base_path, "nvidia-smi.exe")
                    if os.path.exists(full_path):
                        nvidia_smi_cmd = full_path
                        break
                except Exception:
                    continue
        
        # Try direct subprocess call first (list form - more reliable)
        try:
            res = subprocess.run(
                [nvidia_smi_cmd, "--query-gpu=index,name,pci.bus_id,uuid,memory.used,memory.total,"
                 "temperature.gpu,fan.speed,pcie.link.gen.current,pcie.link.width.max,"
                 "utilization.gpu --format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0
            )
        except FileNotFoundError:
            # Fallback to shell=True if list form fails (Windows compatibility)
            res = subprocess.run(
                f'"{nvidia_smi_cmd}" --query-gpu=index,name,pci.bus_id,uuid,memory.used,memory.total,'
                f'temperature.gpu,fan.speed,pcie.link.gen.current,pcie.link.width.max,'
                f'utilization.gpu --format=csv,noheader,nounits',
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0
            )
        
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 12:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus_id": parts[2],
                        "uuid": parts[3],
                        "vram_used_mb": int(parts[4]) if parts[4] else 0,
                        "vram_total_mb": int(parts[5]) if parts[5] else 0,
                        "temp_c": int(parts[6]) if parts[6] else 0,
                        "fan_pct": int(parts[7]) if parts[7] else 0,
                        "pcie_gen": int(parts[8]) if parts[8] else 0,
                        "pcie_width_max": int(parts[9]) if parts[9] else 0,
                        "utilization_gpu": int(parts[10]) if parts[10] else 0
                    })
            if gpus:
                logger.info(f"GPU detection via nvidia-smi: {len(gpus)} GPUs found")
                return gpus
        else:
            logger.debug(f"nvidia-smi returned rc={res.returncode}, stderr={res.stderr[:200] if res.stderr else 'none'}")
    except Exception as e:
        logger.warning(f"nvidia-smi subprocess failed: {e}")
    
    # ============================================================
    # Method 2: Try nvidia-ml-py (pynvml) — pure-Python NVML binding
    # ============================================================
    try:
        import ctypes
        
        # Try loading NVML library with platform-specific paths
        nvml_lib = None
        lib_paths_to_try = []
        
        if _sys.platform == "win32":
            # Windows: nvcuda.dll or explicit NVML path
            for base in [r"C:\Windows\System32", r"C:\Program Files\NVIDIA Corporation"]:
                try:
                    full = os.path.join(base, "nvcuda.dll")
                    if os.path.exists(full):
                        lib_paths_to_try.append(full)
                        break
                except Exception:
                    continue
            # Also try standard library names (Windows may have nvidia-smi driver files)
            for name in ["libnvml.so", "libnvml.so.1", "nvcuda.dll"]:
                lib_paths_to_try.append(name)
        else:
            # Linux/Unix
            for path in ["/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", 
                         "/usr/lib/libnvidia-ml.so.1", "libnvml.so.1", "libnvml.so"]:
                lib_paths_to_try.append(path)
        
        # Try loading each library path
        for lib_path in lib_paths_to_try:
            try:
                nvml_lib = ctypes.CDLL(lib_path)
                logger.debug(f"Successfully loaded NVML from: {lib_path}")
                break
            except Exception:
                continue
        
        if nvml_lib is None:
            raise ImportError("Could not load any NVML library")
        
        # Initialize NVML
        nvml_lib.nvmlInit_v2.restype = ctypes.c_int
        init_ret = nvml_lib.nvmlInit_v2()
        if init_ret != 0:  # NVML_SUCCESS = 0
            raise OSError(f"nvmlInit_v2 failed with code {init_ret}")
        
        # Get device count
        nvml_lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        nvml_lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        count = ctypes.c_uint(0)
        ret = nvml_lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        if ret != 0:
            raise OSError(f"nvmlDeviceGetCount_v2 failed with code {ret}")
        
        # Get properties for each device
        NVML_DEVICE_NAME_MAX = 96
        NVML_STRING_BUFFER_SIZE = 64
        NVML_MAX_GPU_PROPS = 16
        
        for i in range(count.value):
            dev = ctypes.c_void_p()
            
            # Get handle by index
            nvml_lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            nvml_lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            ret = nvml_lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(dev))
            if ret != 0:
                logger.warning(f"nvmlDeviceGetHandleByIndex_v2 failed for device {i}, code {ret}")
                continue
            
            # Get device name
            nvml_lib.nvmlDeviceGetName.restype = ctypes.c_int
            nvml_lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
            name_buf = ctypes.create_string_buffer(NVML_STRING_BUFFER_SIZE)
            ret = nvml_lib.nvmlDeviceGetName(dev, name_buf, NVML_STRING_BUFFER_SIZE)
            device_name = name_buf.value.decode('utf-8', errors='replace') if ret == 0 else f"GPU {i}"
            
            # Get UUID
            nvml_lib.nvmlDeviceGetUUID.restype = ctypes.c_int
            nvml_lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
            uuid_buf = ctypes.create_string_buffer(96)
            ret = nvml_lib.nvmlDeviceGetUUID(dev, uuid_buf, 96)
            device_uuid = uuid_buf.value.decode('utf-8', errors='replace')[:36] if ret == 0 else f"GPU-{i}-uuid"
            
            # Get memory info using nvmlMemory_t struct
            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                ]
            
            nvml_lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(nvmlMemory_t)]
            mem_info = nvmlMemory_t()
            ret = nvml_lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem_info))
            total_mb = int(mem_info.total / (1024 * 1024)) if ret == 0 else 0
            used_mb = int(mem_info.used / (1024 * 1024)) if ret == 0 else 0
            
            # Get temperature
            nvml_lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
            nvml_lib.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            temp_c = ctypes.c_int(0)
            ret = nvml_lib.nvmlDeviceGetTemperature(dev, 0, ctypes.byref(temp_c))
            temperature = temp_c.value if ret == 0 else 0
            
            # Get fan speed
            nvml_lib.nvmlDeviceGetFanSpeed.restype = ctypes.c_int
            nvml_lib.nvmlDeviceGetFanSpeed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            fan_speed = ctypes.c_uint(0)
            ret = nvml_lib.nvmlDeviceGetFanSpeed(dev, ctypes.byref(fan_speed))
            fan_pct = fan_speed.value if ret == 0 else 0
            
            gpus.append({
                "index": i,
                "name": device_name.strip() if device_name.strip() else f"GPU {i}",
                "pci_bus_id": "",  # NVML doesn't provide PCI bus ID directly
                "uuid": device_uuid,
                "vram_used_mb": used_mb,
                "vram_total_mb": total_mb,
                "temp_c": temperature,
                "fan_pct": fan_pct,
                "pcie_gen": 0,
                "pcie_width_max": 0,
                "utilization_gpu": 0
            })
        
        if gpus:
            logger.info(f"GPU detection via pynvml/NVML: {len(gpus)} GPUs found")
            return gpus
        
    except Exception as e:
        logger.warning(f"pynvml/NVML fallback failed: {e}")
    
    # ============================================================
    # Method 3: Try PyTorch CUDA — last resort pure-Python detection
    # ============================================================
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "pci_bus_id": "",
                    "uuid": f"GPU-{i}-torch",
                    "vram_used_mb": 0,
                    "vram_total_mb": int(props.total_mem_mb),
                    "temp_c": 0,
                    "fan_pct": 0,
                    "pcie_gen": 0,
                    "pcie_width_max": 0,
                    "utilization_gpu": 0
                })
            if gpus:
                logger.info(f"GPU detection via PyTorch CUDA: {len(gpus)} GPUs found")
                return gpus
    except Exception as e:
        logger.warning(f"PyTorch CUDA fallback failed: {e}")
    
    # ============================================================
    # Method 4: Try Docker container exec — if vLLM container is running with GPU access
    # ============================================================
    try:
        client = docker.from_env()
        container = client.containers.get(VLLM_CONTAINER_NAME)
        if container.status == "running":
            res = container.exec_run(
                'nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,memory.used,memory.total,'
                'temperature.gpu,fan.speed,pcie.link.gen.current,pcie.link.width.max,'
                'utilization.gpu --format=csv,noheader,nounits'
            )
            if res.exit_code == 0:
                for line in res.output.decode('utf-8', errors='replace').strip().splitlines():
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 12:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "pci_bus_id": parts[2],
                            "uuid": parts[3],
                            "vram_used_mb": int(parts[4]) if parts[4] else 0,
                            "vram_total_mb": int(parts[5]) if parts[5] else 0,
                            "temp_c": int(parts[6]) if parts[6] else 0,
                            "fan_pct": int(parts[7]) if parts[7] else 0,
                            "pcie_gen": int(parts[8]) if parts[8] else 0,
                            "pcie_width_max": int(parts[9]) if parts[9] else 0,
                            "utilization_gpu": int(parts[10]) if parts[10] else 0
                        })
                if gpus:
                    logger.info(f"GPU detection via Docker container exec: {len(gpus)} GPUs found")
                    return gpus
    except Exception as e:
        logger.warning(f"Docker container exec fallback failed: {e}")
    
    # ============================================================
    # Method 5: Try shell=True nvidia-smi (Windows edge case)
    # ============================================================
    try:
        res = subprocess.run(
            f'"{nvidia_smi_cmd}" --query-gpu=index,name,pci.bus_id,uuid,memory.used,memory.total,'
            'temperature.gpu,fan.speed,pcie.link.gen.current,pcie.link.width.max,'
            'utilization.gpu --format=csv,noheader,nounits',
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0
        )
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 12:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus_id": parts[2],
                        "uuid": parts[3],
                        "vram_used_mb": int(parts[4]) if parts[4] else 0,
                        "vram_total_mb": int(parts[5]) if parts[5] else 0,
                        "temp_c": int(parts[6]) if parts[6] else 0,
                        "fan_pct": int(parts[7]) if parts[7] else 0,
                        "pcie_gen": int(parts[8]) if parts[8] else 0,
                        "pcie_width_max": int(parts[9]) if parts[9] else 0,
                        "utilization_gpu": int(parts[10]) if parts[10] else 0
                    })
            if gpus:
                logger.info(f"GPU detection via nvidia-smi shell: {len(gpus)} GPUs found")
    except Exception as e:
        logger.warning(f"nvidia-smi shell fallback failed: {e}")
    
    # Final fallback: return empty list (UI will show "No GPUs detected")
    if not gpus:
        logger.info("GPU detection complete — no GPUs detected via any method")
    else:
        logger.debug(f"Final GPU count: {len(gpus)}")
    
    return gpus

# Periodic history recorder (runs every minute in background)
def _gpu_history_recorder():
    """Background thread that records GPU metrics every 60 seconds."""
    while True:
        time.sleep(60)
        try:
            gpu_data = _parse_nvidia_smi()
            if gpu_data:
                _record_gpu_metrics(gpu_data)
                # Persist to disk every 5 minutes (every 5 recordings)
                if len(gpu_history_buffer["timestamps"]) % 5 == 0:
                    _persist_gpu_history()
        except Exception as e:
            logger.error(f"GPU history recorder error: {e}")

# Start GPU history recorder on startup
def _start_gpu_history_recorder():
    t = threading.Thread(target=_gpu_history_recorder, daemon=True)
    t.start()
    logger.info("GPU history recorder started (records every 60s)")

# Register the recorder to run on app startup
@app.on_event("startup")
def _register_startup_hooks():
    load_last_verified_context()
    load_last_speed_diagnostic()
    t = threading.Thread(target=vllm_container_monitor_loop, daemon=True)
    t.start()
    # Load persisted GPU history and start recorder
    _load_gpu_history()
    _start_gpu_history_recorder()

# ============================================================
# API ENDPOINTS: GPU MONITOR & SELECTION
# ============================================================

@app.get("/api/gpus")
def list_available_gpus():
    """List all available GPUs with PCI Bus ID, PCIe info, and VRAM."""
    gpus = _parse_nvidia_smi()
    
    # Add assignment info if available
    assignment = _get_gpu_assignment()
    tp_slots = assignment.get("tp_slots", [])
    pp_slots = assignment.get("pp_slots", [])
    
    for gpu in gpus:
        pci_id = gpu["pci_bus_id"]
        assigned_to = None
        
        # Check if this GPU is assigned to any TP slot
        for i, slot_pci in enumerate(tp_slots):
            if slot_pci == pci_id:
                assigned_to = f"TP Slot {i}"
                break
        
        if not assigned_to:
            for i, slot_pci in enumerate(pp_slots):
                if slot_pci == pci_id:
                    assigned_to = f"PP Slot {i}"
                    break
        
        gpu["assigned_to"] = assigned_to
    
    return {"gpus": gpus}

@app.get("/api/gpus/monitor")
def get_gpu_monitor_data():
    """Get real-time GPU metrics + 24-hour history for charting."""
    current_gpus = _parse_nvidia_smi()
    
    # Record current metrics into ring buffer
    if current_gpus:
        _record_gpu_metrics(current_gpus)
    
    # Build history payload (only last N points to keep response small)
    n_points = min(len(gpu_history_buffer["timestamps"]), MAX_HISTORY_POINTS)
    timestamps = gpu_history_buffer["timestamps"][-n_points:] if n_points > 0 else []
    
    history_payload = {
        "timestamps": [t for t in timestamps],
        "gpu_vram_used_mb": {},
        "gpu_temp_c": {},
        "gpu_fan_pct": {}
    }
    
    for idx, values in gpu_history_buffer["gpu_vram_used_mb"].items():
        history_payload["gpu_vram_used_mb"][idx] = list(values[-n_points:]) if n_points > 0 else []
    for idx, values in gpu_history_buffer["gpu_temp_c"].items():
        history_payload["gpu_temp_c"][idx] = list(values[-n_points:]) if n_points > 0 else []
    for idx, values in gpu_history_buffer["gpu_fan_pct"].items():
        history_payload["gpu_fan_pct"][idx] = list(values[-n_points:]) if n_points > 0 else []
    
    # Build aggregate stats from current metrics
    total_vram_mb = sum(g.get("vram_total_mb", 0) for g in current_gpus)
    used_vram_mb = sum(g.get("vram_used_mb", 0) for g in current_gpus)
    
    return {
        "current": current_gpus,
        "aggregate": {
            "total_vram_mb": total_vram_mb,
            "used_vram_mb": used_vram_mb,
            "free_vram_mb": total_vram_mb - used_vram_mb
        },
        "history": history_payload
    }

@app.post("/api/gpu/assign")
def assign_gpus(payload: Dict[str, Any]):
    """Assign specific GPUs to TP and PP slots."""
    tp_slots = payload.get("tp_slots", [])
    pp_slots = payload.get("pp_slots", [])
    
    assignment = {
        "tp_slots": tp_slots,
        "pp_slots": pp_slots
    }
    
    _save_gpu_assignment(assignment)
    logger.info(f"GPU assignment saved: TP={tp_slots}, PP={pp_slots}")
    
    return {"message": "GPU assignment saved. Will be applied on next container start.", 
            "tp_slots": tp_slots, "pp_slots": pp_slots}

@app.get("/api/gpu/assignment")
def get_gpu_assignment():
    """Return current GPU slot assignments."""
    assignment = _get_gpu_assignment()
    return assignment

def stop_vllm_container():
    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        logger.info(f"Stopping container {VLLM_CONTAINER_NAME}...")
        container.stop()
        return {"status": "stopping", "message": f"Container {VLLM_CONTAINER_NAME} has been stopped."}
    except Exception as e:
        logger.error(f"Failed to stop container: {e}")
        raise HTTPException(status_code=500, detail=f"Docker error: {str(e)}")

@app.post("/api/container/start")
def start_vllm_container():
    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        logger.info(f"Starting container {VLLM_CONTAINER_NAME}...")
        container.start()
        return {"status": "starting", "message": f"Container {VLLM_CONTAINER_NAME} is starting."}
    except Exception as e:
        logger.error(f"Failed to start container: {e}")
        raise HTTPException(status_code=500, detail=f"Docker error: {str(e)}")

# WebSocket: logs stream
@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    client = docker.from_env()
    try:
        container = client.containers.get(VLLM_CONTAINER_NAME)
        # Fetch existing tail logs and stream new ones
        log_generator = container.logs(stdout=True, stderr=True, stream=True, follow=True, tail=150)
        
        # Make the generator non-blocking or run in separate thread
        loop = asyncio.get_event_loop()
        
        def get_next_log(gen):
            try:
                return next(gen)
            except StopIteration:
                return None
            except Exception as e:
                return str(e).encode('utf-8')
                
        while True:
            # Run the next() call in executor to not block the event loop
            log_line = await loop.run_in_executor(None, get_next_log, log_generator)
            if log_line is None:
                break
            await websocket.send_text(log_line.decode('utf-8', errors='replace'))
    except WebSocketDisconnect:
        logger.info("Logs WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in logs WebSocket: {e}")
        try:
            await websocket.send_text(f"\n[UI Error] Failed to read container logs: {e}\n")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# WebSocket: Interactive Terminal
@app.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket):
    await websocket.accept()
    
    # Create pseudo-terminal (PTY)
    master_fd, slave_fd = pty.openpty()
    
    # Set default size
    s = struct.pack("HHHH", 24, 80, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, s)
    
    # Start shell process (bash)
    p = subprocess.Popen(
        ["/bin/bash"],
        preexec_fn=os.setsid,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd="/app",
        env=os.environ.copy()
    )
    # Close slave descriptor in parent
    os.close(slave_fd)
    
    # Set master descriptor to non-blocking
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    
    # Queues for message coordination
    loop = asyncio.get_event_loop()
    
    async def read_pty():
        try:
            while p.poll() is None:
                # Poll PTY for output
                r, _, _ = select.select([master_fd], [], [], 0.01)
                if master_fd in r:
                    try:
                        data = os.read(master_fd, 4096)
                        if data:
                            await websocket.send_text(data.decode('utf-8', errors='replace'))
                    except BlockingIOError:
                        pass
                await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"PTY read error: {e}")
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
                
    async def write_pty():
        try:
            while p.poll() is None:
                msg = await websocket.receive_text()
                try:
                    # Check if resize message
                    data = json.loads(msg)
                    if data.get("type") == "resize":
                        cols = data.get("cols", 80)
                        rows = data.get("rows", 24)
                        size_struct = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size_struct)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Write standard raw terminal inputs to master fd
                os.write(master_fd, msg.encode('utf-8'))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"PTY write error: {e}")
        finally:
            if p.poll() is None:
                logger.info("Terminating terminal shell process")
                p.terminate()
                p.wait()
                
    await asyncio.gather(read_pty(), write_pty())

# GitHub: Clone/Pull/Browse Repos
@app.get("/api/github/repos")
def list_repos():
    repos = []
    repos_dir = "/repos"
    if not os.path.exists(repos_dir):
        os.makedirs(repos_dir, exist_ok=True)
        
    for entry in os.scandir(repos_dir):
        if entry.is_dir():
            repo_path = entry.path
            git_dir = os.path.join(repo_path, ".git")
            
            # Basic git information
            current_branch = "N/A"
            latest_commit = "N/A"
            pull_time = "Never pulled"
            
            if os.path.exists(git_dir):
                try:
                    # Branch
                    branch_cmd = subprocess.run(
                        ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True, check=True
                    )
                    current_branch = branch_cmd.stdout.strip()
                    
                    # Latest commit
                    commit_cmd = subprocess.run(
                        ["git", "-C", repo_path, "log", "-1", "--format=%h - %an, %ar : %s"],
                        capture_output=True, text=True, check=True
                    )
                    latest_commit = commit_cmd.stdout.strip()
                    
                    # Check modified time of git head to estimate last operation
                    head_file = os.path.join(git_dir, "FETCH_HEAD")
                    if os.path.exists(head_file):
                        mtime = os.path.getmtime(head_file)
                        pull_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                except Exception as e:
                    logger.error(f"Error querying git for {entry.name}: {e}")
                    
            # Check if repo has run_vllm.sh or docker-compose.yml
            has_run_vllm = os.path.exists(os.path.join(repo_path, "run_vllm.sh")) or os.path.exists(os.path.join(repo_path, "models", "run_vllm.sh"))
            has_compose = os.path.exists(os.path.join(repo_path, "docker-compose.yml"))
            
            repos.append({
                "name": entry.name,
                "path": repo_path,
                "branch": current_branch,
                "latest_commit": latest_commit,
                "last_pull": pull_time,
                "has_run_vllm": has_run_vllm,
                "has_compose": has_compose
            })
            
    return repos

def run_git_clone_task(url: str, repo_name: str):
    repos_dir = "/repos"
    dest_path = os.path.join(repos_dir, repo_name)
    
    active_git_operations[repo_name] = {
        "status": "cloning",
        "progress": 0,
        "logs": "",
        "error": None
    }
    
    try:
        process = subprocess.Popen(
            ["git", "clone", url, dest_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            active_git_operations[repo_name]["logs"] += line
            
        rc = process.wait()
        if rc == 0:
            active_git_operations[repo_name]["status"] = "completed"
            active_git_operations[repo_name]["progress"] = 100
        else:
            active_git_operations[repo_name]["status"] = "failed"
            active_git_operations[repo_name]["error"] = f"git clone exited with code {rc}"
    except Exception as e:
        active_git_operations[repo_name] = {
            "status": "failed",
            "progress": 0,
            "logs": str(e),
            "error": str(e)
        }

@app.post("/api/github/clone")
def clone_repo(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Repo URL is required")
        
    url = url.strip()
    # Simple validation
    if not (url.startswith("https://") or url.startswith("git@")):
        raise HTTPException(status_code=400, detail="Invalid git URL. Must start with https:// or git@")
        
    # Derive name
    repo_name_match = re.search(r"/([^/]+?)(?:\.git)?$", url)
    if not repo_name_match:
        raise HTTPException(status_code=400, detail="Could not extract repository name from URL")
        
    repo_name = repo_name_match.group(1)
    repos_dir = "/repos"
    dest_path = os.path.join(repos_dir, repo_name)
    
    if os.path.exists(dest_path):
        raise HTTPException(status_code=400, detail=f"Directory/Repository '{repo_name}' already exists")
        
    background_tasks.add_task(run_git_clone_task, url, repo_name)
    return {"message": "Cloning repository in the background", "repo_name": repo_name}

def run_git_pull_task(repo_name: str):
    repo_path = os.path.join("/repos", repo_name)
    
    active_git_operations[repo_name] = {
        "status": "pulling",
        "progress": 0,
        "logs": "",
        "error": None
    }
    
    try:
        process = subprocess.Popen(
            ["git", "-C", repo_path, "pull"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            active_git_operations[repo_name]["logs"] += line
            
        rc = process.wait()
        if rc == 0:
            active_git_operations[repo_name]["status"] = "completed"
            active_git_operations[repo_name]["progress"] = 100
        else:
            active_git_operations[repo_name]["status"] = "failed"
            active_git_operations[repo_name]["error"] = f"git pull exited with code {rc}"
    except Exception as e:
        active_git_operations[repo_name] = {
            "status": "failed",
            "progress": 0,
            "logs": str(e),
            "error": str(e)
        }

@app.post("/api/github/pull")
def pull_repo(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    repo_name = payload.get("repo_name")
    if not repo_name:
        raise HTTPException(status_code=400, detail="repo_name is required")
        
    repo_path = os.path.join("/repos", repo_name)
    if not os.path.exists(repo_path) or not os.path.exists(os.path.join(repo_path, ".git")):
        raise HTTPException(status_code=404, detail="Git repository folder not found")
        
    background_tasks.add_task(run_git_pull_task, repo_name)
    return {"message": "Pulling updates in the background", "repo_name": repo_name}

@app.get("/api/github/operation/status")
def get_git_status(repo_name: str):
    if repo_name not in active_git_operations:
        return {"status": "idle", "progress": 0, "logs": ""}
    return active_git_operations[repo_name]

@app.get("/api/github/files")
def browse_repo_files(repo_name: str):
    repo_path = os.path.join("/repos", repo_name)
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repository not found")
        
    # Helper to recursively list files in structured JSON
    def list_files_tree(path: str, relative_base: str = "") -> List[Dict[str, Any]]:
        nodes = []
        try:
            for entry in os.scandir(path):
                # Ignore .git folder
                if entry.name == ".git":
                    continue
                    
                rel_path = os.path.join(relative_base, entry.name).replace("\\", "/")
                if entry.is_dir():
                    nodes.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "directory",
                        "children": list_files_tree(entry.path, rel_path)
                    })
                else:
                    nodes.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "file",
                        "size": entry.stat().st_size
                    })
        except Exception as e:
            logger.error(f"Error building file tree for {path}: {e}")
            
        # Sort directories first, then files
        nodes.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        return nodes

    return list_files_tree(repo_path)

@app.get("/api/github/file_content")
def get_repo_file_content(repo_name: str, file_path: str):
    repo_path = os.path.join("/repos", repo_name)
    full_path = os.path.abspath(os.path.join(repo_path, file_path))
    
    # Path traversal validation
    if not full_path.startswith(os.path.abspath(repo_path)):
        raise HTTPException(status_code=403, detail="Access denied (path traversal detected)")
        
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/github/diff_config")
def diff_repo_config(repo_name: str, config_file_path: str):
    """
    Compares a run_vllm.sh or docker-compose.yml file from a repository with our active one.
    """
    repo_path = os.path.join("/repos", repo_name)
    repo_file = os.path.abspath(os.path.join(repo_path, config_file_path))
    
    if not repo_file.startswith(os.path.abspath(repo_path)):
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not os.path.exists(repo_file):
        raise HTTPException(status_code=404, detail="Repository configuration file not found")
        
    # Read repo file
    with open(repo_file, "r", encoding="utf-8", errors="replace") as f:
        repo_content = f.read()
        
    # Determine the corresponding active target file
    is_run_vllm = "run_vllm.sh" in config_file_path
    is_compose = "docker-compose.yml" in config_file_path
    
    active_content = ""
    active_path = ""
    
    if is_run_vllm:
        active_path = RUN_VLLM_PATH
    elif is_compose:
        active_path = "/app/docker-compose.yml" # mapped or local to compose? Actually, it's relative to root.
        # Wait, if compose is not in /app inside this container, we can read /models/../docker-compose.yml since /models is C:\Github Pulls\LM_Studio_Bench\models
        # Thus C:\Github Pulls\LM_Studio_Bench\docker-compose.yml is /models/../docker-compose.yml! That is extremely clever and robust.
        active_path = "/models/../docker-compose.yml"
        
    if active_path and os.path.exists(active_path):
        try:
            with open(active_path, "r", encoding="utf-8", errors="replace") as f:
                active_content = f.read()
        except Exception as e:
            logger.error(f"Could not read active config: {e}")
            
    return {
        "repo_content": repo_content,
        "active_content": active_content,
        "active_path": active_path
    }

@app.post("/api/github/apply_config")
def apply_repo_config(payload: Dict[str, Any]):
    repo_name = payload.get("repo_name")
    config_file_path = payload.get("config_file_path")
    
    if not repo_name or not config_file_path:
        raise HTTPException(status_code=400, detail="repo_name and config_file_path are required")
        
    repo_path = os.path.join("/repos", repo_name)
    repo_file = os.path.abspath(os.path.join(repo_path, config_file_path))
    
    if not repo_file.startswith(os.path.abspath(repo_path)):
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not os.path.exists(repo_file):
        raise HTTPException(status_code=404, detail="Repository file not found")
        
    # Determine the target file
    is_run_vllm = "run_vllm.sh" in config_file_path
    is_compose = "docker-compose.yml" in config_file_path
    
    if is_run_vllm:
        target_path = RUN_VLLM_PATH
    elif is_compose:
        target_path = "/models/../docker-compose.yml"
    else:
        raise HTTPException(status_code=400, detail="Only run_vllm.sh or docker-compose.yml can be applied directly")
        
    try:
        # Create backup
        if os.path.exists(target_path):
            backup_path = f"{target_path}.bak.{int(time.time())}"
            shutil.copy2(target_path, backup_path)
            
        # Write new content
        shutil.copy2(repo_file, target_path)
        if is_run_vllm:
            os.chmod(target_path, 0o755)
            
        return {"message": f"Successfully applied config to {os.path.basename(target_path)}. Run container restart to apply."}
    except Exception as e:
        logger.error(f"Failed to apply config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API: System GPUs
@app.get("/api/system/gpus")
def get_system_gpus():
    gpus = []
    # 1. Try NVIDIA-SMI on host
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0
        )
        if res.returncode == 0:
            for idx, line in enumerate(res.stdout.strip().splitlines()):
                if not line.strip(): continue
                parts = line.split(",")
                name = parts[0].strip()
                vram_mb = int(parts[1].strip())
                gpus.append({"index": idx, "vendor": "NVIDIA", "name": name, "vram_mb": vram_mb})
            return {"success": True, "gpus": gpus, "count": len(gpus)}
    except Exception:
        pass
        
    # 2. Try ROCm-SMI on host (AMD)
    try:
        res = subprocess.run(
            ["rocm-smi", "--showproductname"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0
        )
        if res.returncode == 0:
            names = []
            for line in res.stdout.strip().splitlines():
                if "Card Series" in line or "Product Name" in line or "SKU" in line:
                    names.append(line.strip())
            
            for idx, name in enumerate(names):
                vram_mb = 16384 # default fallback if not found
                gpus.append({"index": idx, "vendor": "AMD", "name": name, "vram_mb": vram_mb})
            if gpus:
                return {"success": True, "gpus": gpus, "count": len(gpus)}
    except Exception:
        pass

    # 3. Check VLLM container nvidia-smi
    try:
        client = docker.from_env()
        container = client.containers.get(VLLM_CONTAINER_NAME)
        if container.status == "running":
            res = container.exec_run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
            if res.exit_code == 0:
                for idx, line in enumerate(res.output.decode('utf-8').strip().splitlines()):
                    if not line.strip(): continue
                    parts = line.split(",")
                    name = parts[0].strip()
                    vram_mb = int(parts[1].strip())
                    gpus.append({"index": idx, "vendor": "NVIDIA", "name": name, "vram_mb": vram_mb})
                return {"success": True, "gpus": gpus, "count": len(gpus)}
    except Exception:
        pass

    return {"success": False, "gpus": [], "count": 0, "error": "No GPUs detected"}

# API: Sampling config GET
@app.get("/api/config/sampling")
def get_sampling_config():
    return load_sampling_params()

# API: Sampling config POST
@app.post("/api/config/sampling")
def save_sampling_config(payload: Dict[str, Any]):
    params = {}
    try:
        params["temperature"] = float(payload.get("temperature", 0.7))
        params["top_p"] = float(payload.get("top_p", 1.0))
        params["top_k"] = int(payload.get("top_k", -1))
        params["presence_penalty"] = float(payload.get("presence_penalty", 0.0))
        params["frequency_penalty"] = float(payload.get("frequency_penalty", 0.0))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter type: {e}")
        
    try:
        os.makedirs(os.path.dirname(SAMPLING_PARAMS_PATH), exist_ok=True)
        with open(SAMPLING_PARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save sampling settings: {str(e)}")
        
    return {"message": "Sampling parameters saved successfully"}

active_system_updates: Dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "logs": "",
    "error": None
}

def run_system_update_task():
    global active_system_updates
    active_system_updates = {
        "status": "updating",
        "progress": 0,
        "logs": "Initializing update process...\n",
        "error": None
    }
    
    if not os.path.exists("/app/.git"):
        active_system_updates["status"] = "failed"
        active_system_updates["error"] = "Not a git repository mount"
        active_system_updates["logs"] += "[Error] /app/.git folder not found. Ensure the host project folder is mounted to /app in docker-compose.yml.\n"
        return
        
    try:
        active_system_updates["logs"] += "Running git pull origin main...\n"
        process = subprocess.Popen(
            ["git", "-C", "/app", "pull", "origin", "main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            active_system_updates["logs"] += line
            
        rc = process.wait()
        if rc == 0:
            active_system_updates["status"] = "completed"
            active_system_updates["progress"] = 100
            active_system_updates["logs"] += "\n✓ System updated successfully from origin main! Please restart the container/service to apply changes.\n"
        else:
            active_system_updates["status"] = "failed"
            active_system_updates["error"] = f"git pull exited with code {rc}"
            active_system_updates["logs"] += f"\n[Error] Update failed with status code {rc}\n"
    except Exception as e:
        active_system_updates["status"] = "failed"
        active_system_updates["error"] = str(e)
        active_system_updates["logs"] += f"\n[Exception] Error during update: {e}\n"

@app.get("/api/version")
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), "version.txt")
    if os.path.exists(version_path):
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                return {"version": f.read().strip()}
        except Exception:
            pass
    return {"version": "1.000"}

@app.post("/api/update")
def start_system_update(background_tasks: BackgroundTasks):
    if active_system_updates["status"] == "updating":
        return {"message": "Update already in progress"}
    background_tasks.add_task(run_system_update_task)
    return {"message": "Update started in background"}

@app.get("/api/update/status")
def get_system_update_status():
    return active_system_updates

# Mount static files and templates
# Wait! Let's ensure directories exist first before mounting
os.makedirs("/app/static", exist_ok=True)
os.makedirs("/app/templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/")
def read_root():
    template_path = "/app/templates/index.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>vLLM Manager UI Server online</h1><p>Frontend template not loaded yet.</p>")
