"""
HTTP Web Server & REST API for KV-Cache Demonstration and Dashboard.
Uses standard Python library http.server with zero external web framework dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import torch

from src.baseline import UncompressedBaselineEngine
from src.streaming_cache import StreamingLLMEngine
from src.sliding_cache import PureSlidingWindowEngine
from src.h2o_cache import H2OCacheEngine
from src.ogc_cache import OGCCacheEngine
from src.metrics import get_device_allocated_bytes, get_process_rss_bytes, theoretical_kv_cache_bytes


class EngineRegistry:
    """Manages active inference engines with lazy loading and memory hygiene."""

    def __init__(self):
        self._current_engine_name: Optional[str] = None
        self._current_engine: Optional[Any] = None

    def get_engine(self, method: str, **kwargs) -> Any:
        # If switching methods, free old engine to conserve memory
        if self._current_engine is not None and self._current_engine_name != method:
            print(f"[EngineRegistry] Evicting previous engine: {self._current_engine_name}")
            del self._current_engine
            self._current_engine = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if self._current_engine is None or self._current_engine_name != method:
            print(f"[EngineRegistry] Initializing engine for method: {method}")
            if method == "full_baseline":
                eng = UncompressedBaselineEngine()
                eng.load_model()
            elif method == "streaming_llm":
                sinks = kwargs.get("sinks", 4)
                window = kwargs.get("window", 256)
                eng = StreamingLLMEngine(sink_tokens=sinks, window_tokens=window)
                eng.load_model()
            elif method == "sliding_window":
                window = kwargs.get("window", 260)
                eng = PureSlidingWindowEngine(window_tokens=window)
                eng.load_model()
            elif method == "h2o":
                heavy = kwargs.get("heavy_hitters", 64)
                window = kwargs.get("window", 196)
                eng = H2OCacheEngine(heavy_hitters=heavy, recent_tokens=window)
                eng.load_model()
            elif method == "ogc_cache":
                sinks = kwargs.get("sinks", 4)
                window = kwargs.get("window", 192)
                chunk_sz = kwargs.get("chunk_size", 64)
                eng = OGCCacheEngine(sink_tokens=sinks, recent_window=window, chunk_size=chunk_sz)
                eng.load_model()
            else:
                raise ValueError(f"Unknown or unsupported method: {method}")

            self._current_engine = eng
            self._current_engine_name = method

        return self._current_engine


GLOBAL_REGISTRY = EngineRegistry()


class DemoServerHandler(BaseHTTPRequestHandler):
    """Custom request handler serving web UI, plots, and inference APIs."""

    # Uses default BaseHTTPRequestHandler logging

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file("web/index.html", "text/html")
        elif path == "/style.css":
            self._serve_file("web/style.css", "text/css")
        elif path == "/app.js":
            self._serve_file("web/app.js", "application/javascript")
        elif path.startswith("/results/"):
            # Serve plots from results/
            filename = os.path.basename(path)
            filepath = os.path.join("results", filename)
            mime, _ = mimetypes.guess_type(filepath)
            self._serve_file(filepath, mime or "image/png")
        elif path == "/api/presets":
            self._handle_get_presets()
        elif path == "/api/benchmarks":
            self._handle_get_benchmarks()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/infer":
            self._handle_post_infer()
        else:
            self.send_error(404, "Endpoint Not Found")

    def _serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def _handle_get_presets(self):
        try:
            with open("data/benchmark_dataset.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
            # Pick a diverse set of 8 presets across context lengths and depths
            presets = [dataset[0], dataset[2], dataset[4], dataset[8], dataset[10], dataset[14], dataset[20], dataset[24]]
            payload = json.dumps(presets).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(500, f"Error loading presets: {e}")

    def _handle_get_benchmarks(self):
        try:
            res: Dict[str, Any] = {}
            if os.path.exists("results/phase8_ablation_results.json"):
                with open("results/phase8_ablation_results.json", "r") as f:
                    res["phase8"] = json.load(f)
            if os.path.exists("results/phase7_comparative_results.json"):
                with open("results/phase7_comparative_results.json", "r") as f:
                    res["phase7"] = json.load(f)
            if os.path.exists("results/phase6_benchmark_results.json"):
                with open("results/phase6_benchmark_results.json", "r") as f:
                    res["phase6"] = json.load(f)

            payload = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(500, f"Error loading benchmarks: {e}")

    def _handle_post_infer(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            req = json.loads(raw_body.decode("utf-8"))

            document = req.get("document", "").strip()
            question = req.get("question", "").strip()
            method = req.get("method", "streaming_llm")
            budget = int(req.get("budget", 260))
            sinks = int(req.get("sinks", 4))
            window = int(req.get("window", 192))
            chunk_size = int(req.get("chunk_size", 64))

            if not document:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Document content is required."}).encode("utf-8"))
                return

            # Format combined prompt if question provided
            if question and "[Inquiry]" not in document:
                prompt = f"{document}\n\n[Inquiry]\n{question}\nAnswer:"
            else:
                prompt = document

            engine = GLOBAL_REGISTRY.get_engine(
                method=method,
                sinks=sinks,
                window=window,
                chunk_size=chunk_size,
                heavy_hitters=budget - window if budget > window else 64,
            )

            res = engine.generate(prompt=prompt, max_new_tokens=35)

            # Extract fields uniformly
            theo_bytes = getattr(res, "compressed_kv_bytes", 0)
            uncomp_bytes = getattr(res, "uncompressed_kv_bytes", 0)
            savings_pct = getattr(res, "kv_cache_savings_pct", 0.0)
            retained_toks = getattr(res, "retained_tokens", getattr(res, "input_tokens", 0))

            dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            peak_alloc = get_device_allocated_bytes(dev) / (1024 * 1024)
            host_rss = get_process_rss_bytes() / (1024 * 1024)

            gen_text = getattr(res, "generated_text", getattr(res, "output_text", ""))
            num_gen = getattr(res, "output_tokens_generated", getattr(res, "generated_tokens", 0))

            response_data = {
                "generated_text": gen_text,
                "input_tokens": res.input_tokens,
                "retained_tokens": retained_toks,
                "output_tokens_generated": num_gen,
                "prefill_time_ms": res.prefill_time_ms,
                "decode_tokens_per_sec": res.decode_tokens_per_sec,
                "uncompressed_kv_bytes": uncomp_bytes,
                "compressed_kv_bytes": theo_bytes,
                "theoretical_kv_bytes": theo_bytes,
                "kv_cache_savings_pct": savings_pct,
                "peak_device_allocated_mib": peak_alloc,
                "host_rss_mib": host_rss,
                "method": method,
            }

            payload = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        except Exception as e:
            err_payload = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_payload)))
            self.end_headers()
            self.wfile.write(err_payload)


def run_server(port: int = 8080, host: str = "0.0.0.0"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, DemoServerHandler)
    print(f"\n{'='*70}")
    print(f"🚀 KV-Cache Demonstration Server Running at http://localhost:{port}")
    print(f"   Mode A: Interactive Inference Demonstration")
    print(f"   Mode B: Research Benchmark Dashboard")
    print(f"{'='*70}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down cleanly.")
        httpd.server_close()
