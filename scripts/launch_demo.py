"""Cross-platform offline presentation server, with an optional local API."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',action='store_true')
    parser.add_argument('--port',type=int,default=3000)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    dist=root/'Interface'/'dist'
    if not (dist/'index.html').is_file():
        parser.error('Build Interface first: npm ci && npm run build')
    backend=None
    with ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(dist))) as server:
        try:
            if args.backend:
                backend=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000'],cwd=root)
            print(f'Offline presentation: http://localhost:{args.port}',flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if backend:
                backend.terminate()
                try: backend.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    backend.kill();backend.wait()


if __name__=='__main__': main()
