# Servers

OpenAI-compatible server entry points are kept here:

- `modal_qwen_openai_server.py`: Modal deployment for Qwen chat completions.
- `gemini_vertex_openai_server.py`: FastAPI wrapper exposing Vertex Gemini via OpenAI-compatible routes.

Used by:

- `scripts/deploy_modal_qwen.sh`
- `scripts/run_google_humaneval_oneshot.sh`
