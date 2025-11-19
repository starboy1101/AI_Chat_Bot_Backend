from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    user_id: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "guest"
    session_id: Optional[str] = None

class CreateChatRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"
