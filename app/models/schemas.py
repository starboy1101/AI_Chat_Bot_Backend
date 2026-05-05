from pydantic import BaseModel
from typing import Optional, Dict, Any

class LoginRequest(BaseModel):
    user_id: str
    password: str 

class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    user_id: str
    password: str    

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None

class UpdateUserInfo(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    email: str
    bio: str | None = None
    location: str | None = None
    website: str | None = None    

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "guest"
    session_id: Optional[str] = None

    # PDF-first support
    attachment: Optional[Dict[str, Any]] = None

class CreateChatRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"
