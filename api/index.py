"""Vercel serverless entrypoint.

Vercel may hand the app either the original path (/api/predict) or the rewritten one
(/api/index), so the route is read from `?r=` first, with the original path as fallback.
"""

import sys
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI

from berlinrentml.api.main import app as rent_api
from berlinrentml.api.main import load_model

load_model()  # serverless: no startup event for mounted apps


class RouteFromQuery:
    """Pure ASGI middleware: set the request path from ?r=<route>."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            route = parse_qs(scope.get("query_string", b"").decode()).get("r", [""])[0]
            path = "/" + route if route else scope["path"].removeprefix("/api")
            scope = {**scope, "path": path or "/", "raw_path": (path or "/").encode(), "root_path": ""}
        await self.app(scope, receive, send)


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(RouteFromQuery)
app.mount("/", rent_api)
