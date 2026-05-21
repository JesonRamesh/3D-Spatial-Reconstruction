#!/usr/bin/env python3
"""
open_viewer.py — One-command standalone 3D viewer launcher.

Usage:
    python open_viewer.py              # port 8080, auto-opens browser
    python open_viewer.py --port 9000
    python open_viewer.py --no-browser
"""

import argparse
import http.server
import os
import socketserver
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
INDEX_PAGE = "app/static/index.html"


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the repo root with CORP/COEP headers for SharedArrayBuffer support."""

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Print only non-asset requests (skip .ply / .splat / .js polls)
        path = args[0] if args else ""
        if not any(path.endswith(ext) for ext in (".splat", ".ply", ".js", ".png", ".css")):
            super().log_message(fmt, *args)


class ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description="Launch the RoboScene+ 3D viewer.")
    ap.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    ap.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)  # serve from repo root so /outputs/ and /app/static/ resolve

    url = f"http://localhost:{args.port}/{INDEX_PAGE}"

    print(f"\n  ⬡  RoboScene+ 3D Viewer")
    print(f"  Serving from : {REPO_ROOT}")
    print(f"  Open at      : {url}")
    print(f"\n  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    with ReuseServer(("", args.port), CORSHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


if __name__ == "__main__":
    main()
