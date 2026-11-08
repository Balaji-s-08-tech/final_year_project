"""Bound request bodies before Starlette/Pydantic parse or allocate JSON objects."""
from starlette.responses import JSONResponse
from experiments.jsonio import JsonLimits, loads

MAX_BODY_BYTES = 2 * 1024 * 1024


class RequestAdmission:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        # Local-operator prototype only: peer, Host and browser Origin are independent.
        # Forwarded headers never grant access; reverse-proxy/multi-user deployment is unsupported.
        from ipaddress import ip_address
        from urllib.parse import urlsplit
        from backend.config import FRONTEND_ORIGIN
        try:
            peer = scope.get('client')
            local = bool(peer) and ip_address(peer[0]).is_loopback
            host = urlsplit('//'+headers.get(b'host',b'').decode('ascii')).hostname
            local_host = host in ('localhost','127.0.0.1','::1')
            origin = headers.get(b'origin')
            allowed_origin = origin is None or origin.decode('ascii') == FRONTEND_ORIGIN
        except (ValueError, UnicodeError):
            local = local_host = allowed_origin = False
        if not (local and local_host and allowed_origin):
            return await JSONResponse({'detail': 'Local operator API only; peer, Host or Origin not authorized'}, status_code=403)(scope, receive, send)
        length = headers.get(b'content-length')
        try:
            if length is not None and (int(length) < 0 or int(length) > MAX_BODY_BYTES):
                return await JSONResponse({'detail': 'Request exceeds 2 MiB byte budget'}, status_code=413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({'detail': 'Invalid content length'}, status_code=400)(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                return await JSONResponse({'detail': 'Request exceeds 2 MiB byte budget'}, status_code=413)(scope, receive, send)
            body.extend(chunk)
            if not message.get('more_body', False):
                break
        if body and headers.get(b'content-type', b'').split(b';')[0].strip().lower() == b'application/json':
            try:
                loads(body, limits=JsonLimits(max_bytes=MAX_BODY_BYTES))
            except (ValueError, TypeError) as exc:
                return await JSONResponse({'detail': f'Invalid bounded JSON: {exc}'}, status_code=422)(scope, receive, send)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        return await self.app(scope, replay, send)
