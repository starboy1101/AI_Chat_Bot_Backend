# Audio Chat Backend (SwarAI)

FastAPI backend for SwarAI, focused on audio-domain chat and requirement collection.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/chats/chat` | SSE character stream (`text/event-stream`, `data: ...`) |
| `POST` | `/chats/chat_stream` | SSE character stream (`text/event-stream`, `data: ...`) |

## Local AI SRS Regeneration

The normal SRS generator remains the first-pass generator. After it creates the DOCX, the chat flow can ask whether to generate a second local AI version and compare both outputs.

For faster SRS regeneration on a Google Colab T4, run `colab_qwen25_srs_api.py` in Colab and configure the backend with the ngrok URL it prints:

```env
COLAB_SRS_BASE_URL=https://your-current-ngrok-url.ngrok-free.app
COLAB_SRS_ENDPOINT=/generate-srs
COLAB_SRS_MODEL=qwen2.5:7b
COLAB_SRS_NUM_CTX=8192
COLAB_SRS_NUM_PREDICT=2500
COLAB_SRS_TIMEOUT_SECONDS=1800
# Optional, if SRS_API_KEY is set in Colab:
COLAB_SRS_API_KEY=shared-secret
```

When `COLAB_SRS_BASE_URL` is present, only the SRS AI regeneration call uses the Colab FastAPI API. Normal chat continues to use the existing backend/Ollama chat path.

If Colab is not configured, the SRS AI regeneration can still fall back to an Ollama-compatible API:

```env
LOCAL_AI_SRS_BASE_URL=http://localhost:11434
LOCAL_AI_SRS_TEXT_MODEL=qwen2.5:7b
LOCAL_AI_SRS_HELPER_MODEL=qwen2.5:7b
LOCAL_AI_SRS_NUM_CTX=8192
LOCAL_AI_SRS_MAX_CONTEXT_CHARS=24000
LOCAL_AI_SRS_TIMEOUT_SECONDS=1800
LOCAL_AI_SRS_KEEP_ALIVE=10m
LOCAL_AI_SRS_NUM_GPU=-1
LOCAL_AI_SRS_MAIN_GPU=0
```

The SRS AI pass uses `qwen2.5:7b` by default and keeps this model selection separate from normal chat. It uses requirement-aware BM25/source excerpt selection before asking the model for structured SRS JSON, then completes any remaining SRS table fields from source context and engineering inference instead of leaving template placeholders.

## SSE Frontend Example

A ready-to-use React example is included at:

- `frontend_sse_chat_example.tsx`

It is wired to `POST /chats/chat_stream` and streams incremental `data: ...` chunks (one character per event).
