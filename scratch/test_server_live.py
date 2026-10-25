import threading
import time
import urllib.request
import json
import sys
sys.path.insert(0, ".")
from src.server import run_server

# Start server in background thread
port = 8899
thread = threading.Thread(target=run_server, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True)
thread.start()
time.sleep(1.0)

# 1. Test GET /
req = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
assert req.status == 200
html = req.read().decode("utf-8")
assert "KV-Cache Squeeze" in html
print("✓ GET / returned 200 and index.html")

# 2. Test GET /style.css
req = urllib.request.urlopen(f"http://127.0.0.1:{port}/style.css")
assert req.status == 200
css = req.read().decode("utf-8")
assert "dark-theme" in css
print("✓ GET /style.css returned 200 and CSS")

# 3. Test GET /api/presets
req = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/presets")
assert req.status == 200
presets = json.loads(req.read().decode("utf-8"))
assert len(presets) > 0
print(f"✓ GET /api/presets returned {len(presets)} presets")

# 4. Test GET /api/benchmarks
req = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/benchmarks")
assert req.status == 200
benchmarks = json.loads(req.read().decode("utf-8"))
assert "phase8" in benchmarks
print(f"✓ GET /api/benchmarks returned phase8 data with {len(benchmarks['phase8'])} configurations")

print("\nALL LIVE SERVER CHECKS PASSED SUCCESSFULLY!")
