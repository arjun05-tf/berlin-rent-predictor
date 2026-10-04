"""Vercel serverless entrypoint: mounts the FastAPI app under /api."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI

from berlinrentml.api.main import app as rent_api
from berlinrentml.api.main import load_model

load_model()  # serverless: no startup event for mounted apps

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", rent_api)
