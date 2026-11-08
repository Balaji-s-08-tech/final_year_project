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
