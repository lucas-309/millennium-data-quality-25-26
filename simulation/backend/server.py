"""Simulation HTTP server — stdlib only.

Serves static frontend files from ../frontend and exposes:
  GET  /api/status    — warmup + readiness state
  POST /api/simulate  — run a backtest with user parameters
  POST /api/warmup    — kick off warmup if not already running

Run: `python -m simulation.backend.server` from the repo root.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from simulation.backend import simulator  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _kick_warmup_async() -> None:
    def _run() -> None:
        try:
            simulator.warmup()
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "not found")
            return
        # Defense: make sure we stay inside FRONTEND_DIR
        try:
            file_path.resolve().relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            self.send_error(403, "forbidden")
            return
        data = file_path.read_bytes()
        ctype = CONTENT_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json(simulator.status())
            return

        if path == "/" or path == "":
            self._send_file(FRONTEND_DIR / "index.html")
            return

        if path.startswith("/api/"):
            self.send_error(404, "unknown api endpoint")
            return

        # Serve static file from frontend dir
        rel = path.lstrip("/")
        self._send_file(FRONTEND_DIR / rel)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"invalid JSON: {exc}"}, status=400)
            return

        if path == "/api/warmup":
            _kick_warmup_async()
            self._send_json(simulator.status())
            return

        if path == "/api/simulate":
            try:
                result = simulator.run_simulation(body)
                self._send_json(result)
            except RuntimeError as exc:
                self._send_json({"error": str(exc), "status": simulator.status()}, status=409)
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": str(exc)}, status=400)
            return

        self.send_error(404, "unknown endpoint")


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    # Kick off warmup in the background so the first /api/simulate call is fast.
    _kick_warmup_async()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"simulation server listening on http://{host}:{port}")
    print(f"  frontend: {FRONTEND_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    main(host=args.host, port=args.port)
