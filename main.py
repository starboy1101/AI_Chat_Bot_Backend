import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from chats import router as chats_router
from config import CORS_ORIGINS
import uvicorn
from backend import load_models_if_needed  
from fastapi.staticfiles import StaticFiles

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

@app.get("/")
def root():
    return {"message": "SwarAI Backend running"}

@app.on_event("startup")
async def startup_event():
    try:
        load_models_if_needed()
        logger.info("Model load scheduled/initiated.")
    except Exception:
        logger.exception("Model load scheduling failed.")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
