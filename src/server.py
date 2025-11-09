#!/usr/bin/env python3
# server.py
import argparse
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import logging
import signal
import ssl

import utils
import routes

def as_json_bytes(obj): 
    return json.dumps(
        obj, ensure_ascii=False, separators=(",", ":")
    ).encode()


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        # CORS + basic security (since we're over HTTPS)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()
    
    def send_json(self, status, obj):
        body = as_json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    
    # Add a tiny health endpoint
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path.startswith("/get_route/") or self.path.startswith("get_route/"):
            try:
                origin, dest = utils.parse_get_route_path(self.path)
                top_routes =  routes.topk_routes(origin, dest)
                
                self.send_json(200, top_routes)
            except ValueError as e:
                self.send_json(400, {"error": str(e)})
            
            return
        else:
            super().do_GET()

    # Cleaner access logs
    def log_message(self, fmt, *args):
        logging.info("%s - - [%s] %s",
                     self.client_address[0],
                     self.log_date_time_string(),
                     fmt % args)
        
    

def main():
    ap = argparse.ArgumentParser(description="Minimal Python HTTPS server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", default="cert.pem")
    ap.add_argument("--key", default="key.pem")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    # Threaded HTTP server (serves files from current working dir)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    # TLS context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    # Safe default ciphers; leave as-is or tweak for your environment
    try:
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!3DES")
    except ssl.SSLError:
        pass  # some older OpenSSL builds might not support this exact string
    context.load_cert_chain(certfile=args.cert, keyfile=args.key)

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    def _shutdown(*_):
        logging.info("Shutting down...")
        httpd.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    logging.info(f"Serving HTTPS on {args.host}:{args.port} (Ctrl+C to quit)")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
