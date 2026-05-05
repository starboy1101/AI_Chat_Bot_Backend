# Audio Chat Backend (SwarAI)

FastAPI backend for SwarAI, focused on audio-domain chat and requirement collection.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/chats/chat` | SSE character stream (`text/event-stream`, `data: ...`) |
| `POST` | `/chats/chat_stream` | SSE character stream (`text/event-stream`, `data: ...`) |

## SSE Frontend Example

A ready-to-use React example is included at:

- `frontend_sse_chat_example.tsx`

It is wired to `POST /chats/chat_stream` and streams incremental `data: ...` chunks (one character per event).
