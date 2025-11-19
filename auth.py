from fastapi import APIRouter, HTTPException
from datetime import datetime
from models import LoginRequest, LoginResponse
from db import supabase, insert_row
from utils import verify_password, hash_password, create_access_token
from config import DEMO_USER, DEMO_PASS
import logging
from uuid import uuid4   

router = APIRouter(prefix="/auth")
logger = logging.getLogger("swarai.auth")

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    # demo shortcut
    # if req.user_id == DEMO_USER and req.password == DEMO_PASS:
    #     token = create_access_token(req.user_id)
    #     return {"success": True, "token": token, "message": "Logged in (demo)"}

    if supabase is None:
        raise HTTPException(status_code=500, detail="Auth backend not configured")

    try:
        r = supabase.table('users').select('*').eq('user_id', req.user_id).limit(1).execute()
        if not r.data:
            raise HTTPException(status_code=401, detail='Invalid credentials')
        user = r.data[0]
        if not verify_password(req.password, user.get('password', '')):
            raise HTTPException(status_code=401, detail='Invalid credentials')
        token = create_access_token(req.user_id)
        return {"success": True, "token": token}
    except HTTPException:
        raise
    except Exception:
        logger.exception('Login error')
        raise HTTPException(status_code=500, detail='Login failed')

@router.post("/register")
async def register(req: LoginRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        now = datetime.utcnow().isoformat()
        hashed = hash_password(req.password)
        insert_row('users', {
            'user_id': req.user_id,
            'password': hashed,
            'created_at': now
        })
        return {"success": True}
    except Exception:
        logger.exception('Register failed')
        raise HTTPException(status_code=500, detail='Register failed')
    
@router.post("/guest")
async def guest_login():
    """
    Create a temporary in-memory guest session.
    No database insert, just returns a short-lived token.
    """
    try:
        guest_id = f"guest_{uuid4().hex[:8]}"
        token = create_access_token(
            guest_id,
            expires_delta=timedelta(hours=2)  # short-lived token
        )
        logger.info(f"Guest session created: {guest_id}")
        return {
            "success": True,
            "guest": True,
            "user_id": guest_id,
            "token": token,
            "message": "Guest session active (temporary)"
        }
    except Exception as e:
        logger.exception("Guest session creation failed")
        raise HTTPException(status_code=500, detail="Guest login failed")
