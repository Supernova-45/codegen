import os
import time
import uuid
from typing import Any

import modal


APP_NAME = os.environ.get("MODAL_APP_NAME", "qwen25-7b-openai")
MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
GPU_TYPE = os.environ.get("MODAL_QWEN_GPU", "L4")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.116.1",
        "pydantic==2.11.7",
        "transformers==4.56.1",
        "torch==2.8.0",
        "accelerate==1.11.0",
        "safetensors==0.6.2",
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=900,
)
@modal.asgi_app()
def qwen_openai_api():
    from fastapi import Body, FastAPI, HTTPException
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    web = FastAPI(title="Qwen2.5-7B OpenAI-Compatible API")

    state: dict[str, Any] = {
        "tokenizer": None,
        "model": None,
        "model_id": MODEL_ID,
    }

    def _ensure_loaded() -> tuple[Any, Any]:
        if state["tokenizer"] is not None and state["model"] is not None:
            return state["tokenizer"], state["model"]

        tokenizer = AutoTokenizer.from_pretrained(state["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            state["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        state["tokenizer"] = tokenizer
        state["model"] = model
        return tokenizer, model

    @web.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state["model_id"],
                    "object": "model",
                    "owned_by": "modal",
                }
            ],
        }

    @web.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        stream = bool(payload.get("stream", False))
        if stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this server")
        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        messages: list[dict[str, str]] = []
        for m in messages_raw:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if isinstance(content, list):
                # Handle OpenAI multi-part content blocks by concatenating text chunks.
                text_parts = [
                    str(chunk.get("text", ""))
                    for chunk in content
                    if isinstance(chunk, dict) and chunk.get("type") == "text"
                ]
                content = "".join(text_parts)
            messages.append({"role": role, "content": str(content)})
        if not messages:
            raise HTTPException(status_code=400, detail="messages must contain at least one valid entry")

        tokenizer, model = _ensure_loaded()
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        max_new_tokens = max(1, min(int(payload.get("max_tokens", 512)), 4096))
        temperature = float(payload.get("temperature", 0.2))
        do_sample = temperature > 0.0

        with torch.inference_mode():
            output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(0.01, temperature) if do_sample else 1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output[0][prompt_ids.shape[-1] :]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True)

        prompt_tokens = int(prompt_ids.shape[-1])
        completion_tokens = int(completion_ids.shape[-1])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model") or state["model_id"]),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return web
