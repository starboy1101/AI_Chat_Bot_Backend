import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, REMEMBER_ME_TOKEN_EXPIRE_DAYS
from app.core.db import insert_row, supabase
from app.models.schemas import LoginRequest, LoginResponse, RegisterRequest
from app.state.chat_state import GUEST_SESSION_TIMEOUT_SECONDS, is_guest_user, touch_guest_user
from app.utils.common import create_access_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/auth")
logger = logging.getLogger("swarai.auth")


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Auth backend not configured")

    try:
        r = supabase.table("users").select("*").eq("user_id", req.user_id).limit(1).execute()
        if not r.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = r.data[0]
        if not verify_password(req.password, user.get("password", "")):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        remember_me = bool(req.remember_me or req.rememberMe)
        expires_delta = None
        if remember_me:
            expires_delta = timedelta(days=REMEMBER_ME_TOKEN_EXPIRE_DAYS)
        token = create_access_token(req.user_id, expires_delta=expires_delta)
        expires_in = int((expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).total_seconds())
        return {
            "success": True,
            "user_id": req.user_id,
            "token": token,
            "remember_me": remember_me,
            "token_expires_in_seconds": expires_in,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Login error")
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/session")
async def get_session(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Login required")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Login required")

    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    return {
        "success": True,
        "user_id": user_id,
        "guest": is_guest_user(user_id),
        "exp": payload.get("exp"),
    }


@router.post("/register")
async def register(req: RegisterRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        existing = supabase.table("users").select("user_id").eq("user_id", req.user_id).limit(1).execute()
        if existing.data:
            return {
                "success": False,
                "message": "User already exists",
            }

        now = datetime.utcnow().isoformat()
        hashed = hash_password(req.password)
        insert_row(
            "users",
            {
                "user_id": req.user_id,
                "email": req.email,
                "first_name": req.firstName,
                "last_name": req.lastName,
                "password": hashed,
                "created_at": now,
            },
        )

        return {"success": True, "message": "User registered successfully"}

    except Exception:
        logger.exception("Register failed")
        raise HTTPException(status_code=500, detail="Register failed")


@router.post("/guest")
async def guest_login():
    try:
        guest_id = f"guest_{uuid4().hex[:8]}"
        touch_guest_user(guest_id)
        token = create_access_token(
            guest_id,
            expires_delta=timedelta(seconds=GUEST_SESSION_TIMEOUT_SECONDS),
        )
        logger.info("Guest session created: %s", guest_id)
        return {
            "success": True,
            "guest": True,
            "user_id": guest_id,
            "token": token,
            "message": "Guest session active (temporary)",
        }
    except Exception:
        logger.exception("Guest session creation failed")
        raise HTTPException(status_code=500, detail="Guest login failed")
