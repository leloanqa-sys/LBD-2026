#!/usr/bin/env python3
"""
Simple zero-dependency HTTP server to host Dapagliflozin LBD Explorer UI.
Run:
    python ui/serve.py
Then open http://localhost:8000 in your browser.
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def run():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"=======================================================")
        print(f" Dapagliflozin LBD Explorer UI running at: {url}")
        print(f" Press Ctrl+C to stop the server.")
        print(f"=======================================================")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run()
