"""Simulation HTTP server — stdlib only.

Serves static frontend files from ../frontend and exposes:
  GET  /api/status    — warmup + readiness state
  POST /api/simulate  — run a backtest with user parameters
  POST /api/warmup    — kick off warmup if not already running

Run: `python -m simulation.backend.server` from the repo root.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def _kick_warmup_async(source: str = None) -> None:
    def _run() -> None:
        try:
            simulator.warmup(source=source)
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()


def _query_source(parsed) -> str | None:
    values = parse_qs(parsed.query).get("source", [])
    return values[0] if values else None


def _validate_source(source: str | None) -> None:
    if source and source not in simulator.SUPPORTED_SOURCES:
        raise ValueError(f"unknown data source: {source!r}")


def _prepare_warmup(source: str | None) -> None:
    if not source:
        return
    with simulator._LOCK:
        current = simulator._STATE.get("data_source")
        if current == source and (simulator._STATE["ready"] or simulator._STATE["loading"]):
            return
        simulator._STATE["ready"] = False
        simulator._STATE["loading"] = False
        simulator._STATE["dataset"] = None
        simulator._STATE["benchmark"] = None
        simulator._STATE["date_min"] = None
        simulator._STATE["date_max"] = None
        simulator._STATE["tickers"] = 0
        simulator._STATE["universe_label"] = None
        simulator._STATE["error"] = None
        simulator._STATE["message"] = f"loading {source} dataset"
        simulator._STATE["data_source"] = source
        simulator._STATE["load_token"] = int(simulator._STATE.get("load_token", 0)) + 1
    simulator._SIM_CACHE.clear()


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
            requested_source = _query_source(parsed)
            try:
                _validate_source(requested_source)
                if requested_source:
                    _prepare_warmup(requested_source)
                    _kick_warmup_async(source=requested_source)
                self._send_json(simulator.status(source=requested_source))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/catalog":
            self._send_json(simulator.catalog())
            return

        if path == "/api/data":
            requested_source = _query_source(parsed)
            try:
                _validate_source(requested_source)
                self._send_json(simulator.data_overview(source=requested_source))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
            return

        if path.startswith("/api/data/ticker/"):
            requested_source = _query_source(parsed)
            tic = path.split("/api/data/ticker/", 1)[1]
            try:
                _validate_source(requested_source)
                self._send_json(simulator.ticker_detail(tic, source=requested_source))
            except ValueError as exc:
                status = 400 if str(exc).startswith("unknown data source") else 404
                self._send_json({"error": str(exc)}, status=status)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
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
            requested_source = body.get("source") if isinstance(body, dict) else None
            try:
                _validate_source(requested_source)
                _prepare_warmup(requested_source)
                _kick_warmup_async(source=requested_source)
                self._send_json(simulator.status(source=requested_source))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/universe/add":
            tickers = body.get("tickers") if isinstance(body, dict) else None
            try:
                result = simulator.add_tickers_to_universe(tickers)
                self._send_json(result)
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": str(exc)}, status=400)
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
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = ap.parse_args()
    main(host=args.host, port=args.port)
