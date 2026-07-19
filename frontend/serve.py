"""Static file server for frontend/ that disables browser caching.

python -m http.server sends no Cache-Control header, so Chrome/Edge can
silently keep serving a stale style.css/index.html from disk cache after
you edit a file, even on a normal refresh. Run this instead during
development so every reload always reflects what's on disk:

    python serve.py [port]   (default port 5500)

Then hard-refresh once (Ctrl+Shift+R) to clear whatever was already cached
from before you started using this server.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    server = ThreadingHTTPServer(('', port), NoCacheHandler)
    print(f'Serving frontend/ with caching disabled on http://localhost:{port}/')
    server.serve_forever()
