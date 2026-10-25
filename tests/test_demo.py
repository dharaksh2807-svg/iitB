"""
Unit and integration tests for Demo CLI and Server APIs.
"""

import json
import subprocess
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.server import DemoServerHandler, GLOBAL_REGISTRY


class TestDemoCLIAndServer(unittest.TestCase):
    """Test suite for demonstration interface and server endpoints."""

    def test_01_cli_dashboard_mode(self):
        """Tests that 'python demo.py --mode dashboard' runs cleanly and outputs metrics."""
        cmd = ["./.venv/bin/python", "demo.py", "--mode", "dashboard"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Dashboard failed: {res.stderr}")
        self.assertIn("Full Baseline", res.stdout)
        self.assertIn("StreamingLLM", res.stdout)
        self.assertIn("OGC-Cache", res.stdout)
        self.assertIn("Granular Depth-Wise Fact Recovery", res.stdout)
        print("  -> test_01_cli_dashboard_mode passed.")

    def test_02_cli_interactive_execution(self):
        """Tests that 'python demo.py --mode interactive --preset 0' executes inference."""
        cmd = [
            "./.venv/bin/python", "demo.py",
            "--mode", "interactive",
            "--preset", "0",
            "--method", "sliding_window"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Interactive demo failed: {res.stderr}")
        self.assertIn("INFERENCE RESULT", res.stdout)
        self.assertIn("Physical KV Cache Size", res.stdout)
        self.assertIn("KV Memory Reduction", res.stdout)
        print("  -> test_02_cli_interactive_execution passed.")

    def test_03_server_static_and_preset_routes(self):
        """Tests that the HTTP server handler serves static assets and API routes."""
        # Test presets endpoint
        handler = MagicMock()
        handler.command = "GET"
        handler.path = "/api/presets"
        handler.headers = {}
        handler.wfile = BytesIO()

        # Call handler method directly
        DemoServerHandler._handle_get_presets(handler)
        handler.send_response.assert_called_with(200)

        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("item_id", data[0])
        print(f"  -> test_03_server_static_and_preset_routes passed ({len(data)} presets returned).")

    def test_04_server_benchmarks_route(self):
        """Tests that /api/benchmarks returns the consolidated benchmark data."""
        handler = MagicMock()
        handler.command = "GET"
        handler.path = "/api/benchmarks"
        handler.headers = {}
        handler.wfile = BytesIO()

        DemoServerHandler._handle_get_benchmarks(handler)
        handler.send_response.assert_called_with(200)

        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertIn("phase8", data)
        self.assertIn("Full Baseline", data["phase8"])
        self.assertIn("OGC-Cache (K=1, C=64)", data["phase8"])
        print("  -> test_04_server_benchmarks_route passed.")


if __name__ == "__main__":
    unittest.main()
