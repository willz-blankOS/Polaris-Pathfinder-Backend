#!/usr/bin/env python3
# server.py
import argparse
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import logging
import signal
import ssl
from urllib.parse import urlparse, parse_qs

import utils
import routes

def as_json_bytes(obj): 
    return json.dumps(
        obj, ensure_ascii=False, separators=(",", ":")
    ).encode()


class ApiHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # CORS + basic security (since we're over HTTPS)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()
    
    def log_message(self, fmt, *args):
        logging.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), fmt % args)
    
    def send_json(self, status, obj):
        body = as_json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self.end_headers()
    
    # Add a tiny health endpoint
    def do_GET(self):
        parsed = urlparse(self.path)
        path, q = parsed.path, {
            k: (v if len(v)>1 else v[0]) 
            for k,v in parse_qs(parsed.query).items()
        }
        
        if path == "/api/ping":
            # Simple JSON response
            return self.send_json(200, {"ok": True, "service": "demo", "version": "1.0"})
        elif path == "/api/echo":
            # Echo query params + client info
            return self.send_json(200, {"query": q, "ip": self.client_address[0]})
        elif path.startswith("/get_route/") or self.path.startswith("get_route/"):
            try:
                origin, dest = utils.parse_get_route_path(path)
                top_routes =  routes.topk_routes(origin, dest)
                
                self.send_json(200, top_routes)
            except ValueError as e:
                self.send_json(400, {"error": str(e)})
            
            return
        else:
            self.send_json(
                404, {"error": "Not found", "path": path}
            )
        
def main():
    ap = argparse.ArgumentParser(description="Minimal Python HTTPS server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", default="cert.pem")
    ap.add_argument("--key", default="key.pem")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!3DES")
    except ssl.SSLError:
        pass
    ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    def shutdown(*_): logging.info("Shutting down…"); httpd.shutdown()
    signal.signal(signal.SIGINT, shutdown)
    try: signal.signal(signal.SIGTERM, shutdown)
    except Exception: pass

    logging.info(f"Serving HTTPS on {args.host}:{args.port}")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
