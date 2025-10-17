from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="GeoExtract API")
app.include_router(router)
