"""
Google Colab FastAPI service for SRS generation with Qwen2.5 7B on a T4 GPU.

In a Colab notebook:
1. Runtime -> Change runtime type -> T4 GPU
2. Run:
   !pip install -q fastapi uvicorn pyngrok transformers accelerate bitsandbytes
3. Optional:
   import os
   os.environ["NGROK_AUTHTOKEN"] = "your-ngrok-token"
   os.environ["SRS_API_KEY"] = "shared-secret"
4. Paste/run this file's code in a cell.
5. Copy the printed ngrok URL into your backend as COLAB_SRS_BASE_URL.
"""

from __future__ import annotations

import os
import re
import threading
import gc
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = os.getenv("COLAB_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("SRS_API_KEY", "").strip()
GPU_MAX_MEMORY = os.getenv("GPU_MAX_MEMORY", "10GiB")
CPU_MAX_MEMORY = os.getenv("CPU_MAX_MEMORY", "24GiB")
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/content/qwen_offload")

app = FastAPI(title="Qwen2.5 7B SRS API")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    max_memory={0: GPU_MAX_MEMORY, "cpu": CPU_MAX_MEMORY},
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:7b"
    temperature: float = 0.2
    num_predict: Optional[int] = 2500
    num_ctx: Optional[int] = 8192
    format: Optional[str] = "json"


def _authorize(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid SRS API key.")


def _strip_prompt_echo(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt) :]
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate-srs")
def generate_srs(req: GenerateRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, str]:
    _authorize(authorization)
    messages = [
        {
            "role": "system",
            "content": "You generate complete SRS JSON. Return only valid JSON and never copy template placeholders.",
        },
        {"role": "user", "content": req.prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_new_tokens = int(req.num_predict or 2500)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=req.temperature > 0,
            temperature=max(float(req.temperature), 0.01),
            pad_token_id=tokenizer.eos_token_id,
        )

    input_token_count = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_token_count:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"text": _strip_prompt_echo(text, prompt), "model": req.model}


if __name__ == "__main__":
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if token:
        ngrok.set_auth_token(token)
    ngrok.kill()
    public_url = ngrok.connect(PORT, "http").public_url
    print(f"COLAB_SRS_BASE_URL={public_url}")
    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info"),
        daemon=True,
    )
    thread.start()
    thread.join()
