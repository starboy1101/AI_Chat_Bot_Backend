import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.chats import router as chats_router
from app.api.routes.srs import router as srs_router
from app.core.config import CORS_ORIGINS
from app.services.backend import preload_models_for_startup

logger = logging.getLogger("swarai.main")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SwarAI Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGINS] if isinstance(CORS_ORIGINS, str) else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(srs_router)


@app.get("/")
def root():
    return {"message": "SwarAI Backend running"}


@app.on_event("startup")
async def startup_event():
    if os.getenv("SKIP_MODEL_PRELOAD_ON_STARTUP", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Skipping model preload on startup because SKIP_MODEL_PRELOAD_ON_STARTUP is enabled.")
        return
    try:
        await preload_models_for_startup()
        logger.info("Model preload completed.")
    except Exception:
        logger.exception("Model preload failed.")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
