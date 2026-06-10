import urllib.request, json, sys

base = "http://localhost:8888"

# Test DELETE model
print("--- DELETE MODEL ---")
try:
    data = json.dumps({"path": "/vllm-cache/nonexistent", "location": "cache"}).encode()
    req = urllib.request.Request(base + "/api/models/delete", data=data, headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req)
    print(f"DELETE: {resp.read().decode()}")
except Exception as e:
    print(f"DELETE ERROR: {e}")

# Test GPU detection
print()
print("--- GPU DETECTION ---")
try:
    req = urllib.request.Request(base + "/api/system/gpus")
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode())
    print(f'GPU count: {data["count"]}')
    for g in data.get("gpus", []):
        print(f'  GPU: {g["name"]} VRAM={g["vram_mb"]}MB Vendor={g["vendor"]}')
except Exception as e:
    print(f"GPU ERROR: {e}")

# Test remaining endpoints
print()
print("--- REMAINING ENDPOINTS ---")

endpoints = [
    ("GET /api/version", "GET", None),
    ("POST /api/container/start", "POST", None),
    ("POST /api/container/stop", "POST", None),
    ("POST /api/container/restart", "POST", None),
    ("GET /api/config/backups", "GET", None),
    ("GET /api/config/sampling", "GET", None),
]

for name, method, body in endpoints:
    try:
        path = "/" + name.split()[1]  # e.g. "/api/version"
        if body is not None and method == "POST":
            data = json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method=method)
        else:
            req = urllib.request.Request(base + path)
        resp = urllib.request.urlopen(req)
        result = resp.read().decode()[:80]
        print(f"  {name}: OK ({result})")
    except Exception as e:
        err = str(e)[:80]
        print(f"  {name}: FAIL ({err})")
