from __future__ import annotations

import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from google import genai
from google.genai import types


load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_VERTEX_MODEL", "gemini-3-pro-preview")
DEFAULT_LOCATION = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
_default_thinking_budget_raw = os.environ.get("GEMINI_VERTEX_THINKING_BUDGET", "").strip()
DEFAULT_THINKING_BUDGET = int(_default_thinking_budget_raw) if _default_thinking_budget_raw else None
DEFAULT_INCLUDE_THOUGHTS = os.environ.get("GEMINI_VERTEX_INCLUDE_THOUGHTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _project_name() -> str:
    project = os.environ.get("GCP_PROJECT_NAME") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Missing GCP_PROJECT_NAME (or GOOGLE_CLOUD_PROJECT). "
            "Set it in .env or your shell before starting the Gemini proxy."
        )
    return project


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(str(chunk.get("text", "")))
        return "".join(text_parts)
    return str(content)


def _convert_messages(messages_raw: list[dict[str, Any]]) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for message in messages_raw:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower()
        text = _normalize_content(message.get("content", ""))
        if role == "system":
            if text:
                system_parts.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))

    system_instruction = "\n\n".join(system_parts).strip() or None
    return system_instruction, contents


def _extract_finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "stop"
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return "stop"
    name = getattr(finish_reason, "name", None)
    return str(name or finish_reason).lower()


def _usage_counts(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    completion_tokens = max(0, total_tokens - prompt_tokens)
    return prompt_tokens, completion_tokens


def _resolve_thinking_budget(payload: dict[str, Any]) -> int | None:
    raw = payload.get("thinking_budget", DEFAULT_THINKING_BUDGET)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_THINKING_BUDGET


def _is_zero_thinking_budget_unsupported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "thinking_budget" in msg and "to 0" in msg and "does not support" in msg


def create_app() -> FastAPI:
    web = FastAPI(title="Vertex Gemini OpenAI-Compatible API")
    state: dict[str, Any] = {"client": None}

    def _ensure_client() -> genai.Client:
        client = state.get("client")
        if client is None:
            state["client"] = genai.Client(
                vertexai=True,
                project=_project_name(),
                location=os.environ.get("GEMINI_VERTEX_LOCATION", DEFAULT_LOCATION),
            )
        return state["client"]

    @web.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": os.environ.get("GEMINI_VERTEX_MODEL", DEFAULT_MODEL),
                    "object": "model",
                    "owned_by": "google-vertex",
                }
            ],
        }

    @web.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if bool(payload.get("stream", False)):
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if int(payload.get("n", 1)) != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")

        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        system_instruction, contents = _convert_messages(messages_raw)
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="messages must include at least one non-system message",
            )

        model_name = str(payload.get("model") or os.environ.get("GEMINI_VERTEX_MODEL", DEFAULT_MODEL))
        config_kwargs: dict[str, Any] = {
            "temperature": float(payload.get("temperature", 0.2)),
            "candidate_count": 1,
        }
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max(1, int(max_tokens))
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        thinking_budget = _resolve_thinking_budget(payload)
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=DEFAULT_INCLUDE_THOUGHTS,
                thinking_budget=thinking_budget,
            )

        client = _ensure_client()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            # Some Gemini models reject thinking_budget=0. Retry without thinking_config
            # so benchmark runs continue instead of hard failing.
            if thinking_budget == 0 and _is_zero_thinking_budget_unsupported(exc):
                retry_kwargs = dict(config_kwargs)
                retry_kwargs.pop("thinking_config", None)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**retry_kwargs),
                    )
                except Exception as retry_exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini API error: {retry_exc}",
                    ) from retry_exc
            else:
                raise HTTPException(status_code=500, detail=f"Gemini API error: {exc}") from exc

        text = response.text or ""
        prompt_tokens, completion_tokens = _usage_counts(response)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _extract_finish_reason(response),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return web


app = create_app()
