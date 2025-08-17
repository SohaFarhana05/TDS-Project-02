# Data Analyst Agent API

Minimal, production-ready FastAPI scaffold for the Data Analyst Agent. This implements:
- A secure HTTP POST ingest/ack endpoint at `/` (multipart questions.txt + attachments) with heuristics and planning scaffold.
- A lightweight analysis and Q&A endpoint at `/api` for simple math/knowledge or small data analysis with pandas/matplotlib and optional LLM routing.

Non-goals: No heavy analysis yet. No model weights included.

## Why these dependencies

- fastapi — lightweight, modern web framework
- uvicorn[standard] — ASGI server for local/dev runs
- python-multipart — parse multipart/form-data uploads
- requests — outbound HTTP for providers and scraping
- beautifulsoup4 — basic HTML parsing for lightweight scraping
- pandas — tabular data wrangling (CSV/Parquet)
- matplotlib — simple plotting when needed
- python-dotenv — load .env for local development
- openai — optional client for OpenAI-hosted inference
- duckdb — in-process SQL/Parquet/CSV query engine

Keep dependencies lean to stay under Vercel free tier limit (~200 MB unpacked). Avoid heavy ML stacks here.

## Environment Variables

Core limits and toggles:
- PER_FILE_MAX_BYTES — per-file upload cap (default 10 MB)
- TOTAL_MAX_BYTES — total upload cap (default 100 MB)
- REQUEST_TIMEOUT_SECONDS — processing timeout (default 170 s)
- SKIP_LLM — if "true", no network model calls; a deterministic heuristic planner is used

LLM provider selection (defaults safe to none):
- LLM_PROVIDER — one of: local, openai_api, huggingface, replicate, none (default: none)
- GPT_OSS_MODEL — model name/slug (default: gpt-oss-20b)
- LOCAL_LLM_ENDPOINT — required if LLM_PROVIDER=local, e.g. http://localhost:8080/generate
- OPENAI_API_KEY — required if LLM_PROVIDER=openai_api
- HUGGINGFACE_API_KEY — required if LLM_PROVIDER=huggingface
- REPLICATE_API_TOKEN — required if LLM_PROVIDER=replicate

Example .env for local dev:
```
# core caps
PER_FILE_MAX_BYTES=10485760
TOTAL_MAX_BYTES=104857600
REQUEST_TIMEOUT_SECONDS=170

# LLM routing
LLM_PROVIDER=none
GPT_OSS_MODEL=gpt-oss-20b
# LOCAL_LLM_ENDPOINT=http://localhost:8080/generate
# OPENAI_API_KEY=sk-...
# HUGGINGFACE_API_KEY=hf_...
# REPLICATE_API_TOKEN=r8_...

# deterministic local plan/testing
SKIP_LLM=true
```

## Run locally

1) Python 3.11+
2) Install deps:
```
pip install -r requirements.txt
```
3) Start the server:
```
uvicorn api.index:app --port 8000 --host 0.0.0.0
```
4) Health check: GET http://localhost:8000/

## API

- POST `/` — multipart/form-data with at least questions.txt (UTF-8). Returns 202 with acknowledgment JSON (scaffolding).
- POST `/api` — multipart/form-data for lightweight Q&A or small data analysis:
  - Required: questions.txt
  - Optional: attachments (CSV/JSON/TXT/others). CSVs are read with pandas; small preview-only for other types. If plotting is requested, a tiny PNG is returned as base64.
  - If the prompt is a simple math expression, it’s answered directly without LLM.
  - Otherwise, if LLM is enabled, a short JSON answer is requested from the configured provider using `GPT_OSS_MODEL` (default `gpt-oss-20b`). No large weights are loaded in-process on Vercel; calls are lazy/outbound.
  - Always returns short JSON. Errors use `{ "error": "message" }`.

Notes on model usage and efficiency:
- The server does not load any large model weights in-process. When `LLM_PROVIDER=local`, it calls your local server endpoint (you host the model). When using `openai_api`, `huggingface`, or `replicate`, calls are made over HTTP.
- Prompts are kept short and direct, and responses are parsed as strict JSON.

## Quick tests (curl)

Simple math (returns {"answer": 4}):
```
# Linux/macOS
curl -s -X POST http://localhost:8000/api \
  -F "questions.txt=@-;type=text/plain" <<< "2 + 2"

# Windows PowerShell
$body = @{
  "questions.txt" = Get-Item "questions.txt"
}
curl.exe -s -X POST http://localhost:8000/api -F "questions.txt=@questions.txt;type=text/plain"
```

Data analysis (attach CSV, ask for summary or plot):
```
curl -s -X POST http://localhost:8000/api \
  -F "questions.txt=@questions.txt;type=text/plain" \
  -F "data.csv=@data.csv;type=text/csv"
```
Where `questions.txt` might contain: `Please provide a summary and a small plot.`

LLM-backed short answer (requires provider env var & token):
```
curl -s -X POST http://localhost:8000/api \
  -F "questions.txt=@-;type=text/plain" <<< "What is the capital of France?"
```

## Vercel deployment notes

- Route to `api/index` via `vercel.json`. Vercel’s Python/ASGI runtime uses the FastAPI app defined at `api/index.py`.
- Configure env vars in Vercel dashboard: LLM_PROVIDER (default none), GPT_OSS_MODEL (`gpt-oss-20b`), provider tokens as needed, SKIP_LLM, and size/timeout caps.
- Keep total installed size under ~200 MB. The chosen packages are pure-Python or small wheels where available.

## Security & privacy

- Strict size caps and timeout to prevent abuse.
- Logs are minimal: IDs, filenames, counts, sizes, provider/model, timestamps.
- Never transmit raw attachment bytes to external services; only metadata and small previews.

## License

MIT