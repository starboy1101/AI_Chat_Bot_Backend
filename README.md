---
title: Audio Chat Backend
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: server.py
pinned: false
---

# 🎧 Audio Chat Backend (SwarAI)

This Hugging Face Space hosts the **FastAPI backend** for the SwarAI audio assistant —  
an intelligent agent specialized in **Windows Audio Architecture**, including:

- 🪟 **WASAPI** (Windows Audio Session API)  
- 🎚️ **IAudioClient / IAudioClient3** interfaces  
- 🎛️ **Audio Processing Objects (APOs)**  
- 🔉 Audio driver frameworks and DSP-level optimization  

---

## 🚀 Features

- **Chat Endpoint (`/chat`)**  
  Handles structured conversations and natural-language questions about audio APIs and system design.  

- **FAISS + Supabase Integration**  
  Stores vectorized Q&A history and retrieves relevant context to improve accuracy.  

- **Local or Remote LLM Support**  
  Uses `llama-cpp-python` with `.gguf` models hosted on Hugging Face Hub  
  (`Omkar1803/mistral-7b-gguf`).  

- **CORS-enabled FastAPI Server**  
  Ready to connect directly with your React/Tailwind frontend chat UI.

---

## 🧩 Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET`  | `/` | Health check – returns `"✅ Audio Chatbot API is running!"` |
| `POST` | `/chat` | Main endpoint – accepts a JSON body: <br> `{ "message": "your query", "user_id": "optional_id" }` |

Example `curl` test:
```bash
curl -X POST "https://Omkar1803-audio-chat-backend.hf.space/chat" \
-H "Content-Type: application/json" \
-d "{\"message\": \"What is IAudioClient?\", \"user_id\": \"test_user\"}"
