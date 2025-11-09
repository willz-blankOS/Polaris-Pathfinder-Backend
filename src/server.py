#!/usr/bin/env python3
# json_api_server.py
import argparse, json, logging, ssl, sys, signal
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# If you have your own utils/routes, import and use them:
import utils, routes

def as_json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

class ApiHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def log_message(self, fmt, *args):
        logging.info("%s - - [%s] %s",
                     self.client_address[0], self.log_date_time_string(), fmt % args)

    # ---- this sends JSON bytes back to the client ----
    def send_json(self, status: int, obj: dict):
        body = as_json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)  # <-- actual payload write

    def do_OPTIONS(self):
        self.send_response(204); self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        query  = {k: (v if len(v)>1 else v[0]) for k,v in parse_qs(parsed.query).items()}

        if path == "/api/ping":
            return self.send_json(200, {"ok": True, "service": "polaris", "version": "1.0"})

        if path == "/api/echo":
            return self.send_json(200, {"query": query, "ip": self.client_address[0]})

        # Example: /api/get_route/from=43.651,-79.383&&to=43.700,-79.400
        if path.startswith("/api/get_route/") or path.startswith("/get_route/"):
            try:
                origin, dest = utils.parse_get_route_path(path)
                result = routes.topk_routes(origin, dest)
                return self.send_json(200, result)
            except Exception as e:
                return self.send_json(400, {"error": f"bad route path: {e}"})

        return self.send_json(404, {"error": "Not Found", "path": path})

def main():
    ap = argparse.ArgumentParser(description="Minimal HTTPS JSON API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", default="cert.pem")
    ap.add_argument("--key", default="key.pem")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    try: ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!3DES")
    except ssl.SSLError: pass
    ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    def _term(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    logging.info(f"Serving HTTPS on {args.host}:{args.port}")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logging.info("Shutting down…")
    finally:
        httpd.server_close()
        logging.info("Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
