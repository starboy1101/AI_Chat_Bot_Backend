import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.core.db import insert_row, supabase
from app.models.schemas import LoginRequest, LoginResponse, RegisterRequest
from app.utils.common import create_access_token, hash_password, verify_password

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
        token = create_access_token(req.user_id)
        return {"success": True, "token": token}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Login error")
        raise HTTPException(status_code=500, detail="Login failed")


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
        token = create_access_token(
            guest_id,
            expires_delta=timedelta(hours=2),
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
