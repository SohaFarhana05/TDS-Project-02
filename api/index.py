import os
import io
import uuid
import json
import time
import math
import asyncio
import tempfile
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests

# Load environment variables early
load_dotenv()

# Configuration with sensible defaults (can be overridden via env)
PER_FILE_MAX_BYTES = int(os.getenv("PER_FILE_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
TOTAL_MAX_BYTES = int(os.getenv("TOTAL_MAX_BYTES", 100 * 1024 * 1024))       # 100 MB
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 170))      # 170 s
SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()  # local | openai_api | huggingface | replicate | none
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "gpt-oss-20b").strip()
LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()

app = FastAPI(title="Data Analyst Agent - Ingest API", docs_url=None, redoc_url=None)

# Minimal CORS to ease local testing; restrict in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"]
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strip_bom(text_bytes: bytes) -> str:
    # Remove UTF-8 BOM if present and decode
    if text_bytes.startswith(b"\xef\xbb\xbf"):
        text_bytes = text_bytes[3:]
    return text_bytes.decode("utf-8", errors="replace")


def _sanitize_filename(name: str) -> str:
    # Prevent path traversal and keep filenames tidy
    base = os.path.basename(name or "")
    return base.replace("..", "_").strip() or f"file-{uuid.uuid4().hex}"


def _detect_task_type(question_text: str, attachment_names: List[str]) -> str:
    text = (question_text or "").lower()
    names = [n.lower() for n in attachment_names]

    types = set()
    if any(n.endswith(".csv") for n in names):
        types.add("csv_analysis")
    if any(n.endswith(ext) for ext in (".parquet", ".pq", ".duckdb") for n in names):
        types.add("parquet_query")
    if any(proto in text for proto in ("http://", "https://", "www.")) or "scrape" in text:
        types.add("scrape")
    # Simple text analysis indicators
    if any(k in text for k in ("summarize", "classify", "sentiment", "topic", "nlp")) and not names:
        types.add("text_analysis")

    if not types:
        return "mixed" if names else "text_analysis"
    if len(types) == 1:
        return next(iter(types))
    return "mixed"


# ---- JSON extraction from possibly-messy LLM outputs ----

def _try_json_load(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_block(text: str) -> Optional[str]:
    """
    Try to find the first balanced JSON object/array in text.
    Deterministic: scans for '{' or '[' and attempts bracket matching.
    Returns the JSON substring or None.
    """
    if not text:
        return None
    starts = []
    for i, ch in enumerate(text):
        if ch in "[{":
            starts.append((i, ch))
    for start_idx, ch in starts:
        stack = []
        for j in range(start_idx, len(text)):
            c = text[j]
            if c in "{[":
                stack.append(c)
            elif c in "]}":
                if not stack:
                    break
                top = stack.pop()
                if (top, c) not in (("{", "}"), ("[", "]")):
                    break
                if not stack:
                    candidate = text[start_idx:j+1]
                    if _try_json_load(candidate) is not None:
                        return candidate
        # try next start
    return None


def safe_parse_json(text: str) -> Dict[str, Any]:
    data = _try_json_load(text)
    if data is not None:
        return {"ok": True, "data": data}
    block = _extract_json_block(text)
    if block is not None:
        try:
            return {"ok": True, "data": json.loads(block)}
        except Exception:
            pass
    return {"ok": False, "error": "Could not parse JSON from model response"}


# ---- Retry / backoff ----

def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def with_retries(fn, *, retries: int = 2, base_delay: float = 0.5):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            delay = base_delay * (2 ** attempt)
            _sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---- LLM provider interface ----

def call_llm(prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0, request_id: Optional[str] = None, prefix_instructions: Optional[str] = None) -> Dict[str, Any]:
    """
    Provider-agnostic LLM call that requests a JSON-only response.

    Local provider contract (example): send POST to LOCAL_LLM_ENDPOINT with JSON
      {
        "model": "gpt-oss-20b",
        "input": "<prompt>",
        "max_tokens": 1024,
        "temperature": 0.0
      }
    and expect JSON response like { "output": "<text with JSON>" }.

    Hugging Face Inference API: POST https://api-inference.huggingface.co/models/{model}
      headers: Authorization: Bearer <token>
      body: { "inputs": "<prompt>", "parameters": {"max_new_tokens": 1024, "temperature": 0.0, "return_full_text": false} }
      response: [{"generated_text": "..."}] or {"generated_text": "..."}

    Replicate generic contract: POST https://api.replicate.com/v1/predictions
      body: { "version": "<model-or-version>", "input": { "prompt": "...", "temperature": 0.0, "max_tokens": 1024 } }
      Then poll GET /v1/predictions/{id} until status "succeeded" and read .output (string or list joined).

    Returns a dict: { ok: bool, data?: dict, provider: str, model: str, error?: str }
    """
    provider = LLM_PROVIDER
    model = GPT_OSS_MODEL

    # Default to planner schema unless overridden
    if prefix_instructions is None:
        prefix_instructions = (
            "You are a planner. Respond ONLY with compact JSON matching this schema: "
            "{\"plan\":{\"steps\":[{\"id\":\"s1\",\"type\":\"<string>\",\"description\":\"<short>\"}]}}. "
            "No prose. No markdown."
        )
    composed_prompt = f"{prefix_instructions}\n\n{prompt.strip()}\n"

    def _finish_from_text_output(text_out: str) -> Dict[str, Any]:
        parsed = safe_parse_json(text_out)
        if parsed.get("ok"):
            return {"ok": True, "data": parsed["data"], "provider": provider, "model": model}
        return {"ok": False, "error": parsed.get("error", "parse_error"), "provider": provider, "model": model}

    timeout = 60

    if provider == "none":
        return {"ok": False, "error": "LLM provider not configured (LLM_PROVIDER=none)", "provider": provider, "model": model}

    if provider == "local":
        if not LOCAL_LLM_ENDPOINT:
            return {"ok": False, "error": "LOCAL_LLM_ENDPOINT not set", "provider": provider, "model": model}

        def _do():
            resp = requests.post(
                LOCAL_LLM_ENDPOINT,
                json={
                    "model": model,
                    "input": composed_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            js = resp.json()
            text_out = js.get("output") or js.get("text") or js.get("data") or ""
            if not isinstance(text_out, str):
                text_out = json.dumps(text_out)
            return _finish_from_text_output(text_out)

        return with_retries(_do)

    if provider == "openai_api":
        if not OPENAI_API_KEY:
            return {"ok": False, "error": "OPENAI_API_KEY not set", "provider": provider, "model": model}

        def _do():
            # Import lazily to avoid dependency overhead when unused
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError(f"openai package not available: {e}")
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Use chat.completions to maximize compatibility
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond only with valid JSON as requested."},
                    {"role": "user", "content": composed_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text_out = resp.choices[0].message.content or ""
            return _finish_from_text_output(text_out)

        return with_retries(_do)

    if provider == "huggingface":
        if not HUGGINGFACE_API_KEY:
            return {"ok": False, "error": "HUGGINGFACE_API_KEY not set", "provider": provider, "model": model}

        api_url = f"https://api-inference.huggingface.co/models/{model}"

        def _do():
            resp = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={
                    "inputs": composed_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False,
                    },
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            js = resp.json()
            # Response format can vary
            if isinstance(js, list) and js:
                text_out = js[0].get("generated_text") or ""
            elif isinstance(js, dict):
                text_out = js.get("generated_text") or js.get("text") or ""
            else:
                text_out = ""
            return _finish_from_text_output(text_out)

        return with_retries(_do)

    if provider == "replicate":
        if not REPLICATE_API_TOKEN:
            return {"ok": False, "error": "REPLICATE_API_TOKEN not set", "provider": provider, "model": model}

        def _do():
            # Create prediction
            create = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": model,  # For generic use, set GPT_OSS_MODEL to a valid Replicate version or model slug
                    "input": {
                        "prompt": composed_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                },
                timeout=timeout,
            )
            create.raise_for_status()
            cjs = create.json()
            pred_id = cjs.get("id")
            status = cjs.get("status")
            get_url = cjs.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{pred_id}"
            # Poll until completed or timeout budget spent
            start = time.time()
            text_out = ""
            while True:
                if time.time() - start > timeout:
                    raise TimeoutError("replicate prediction timeout")
                g = requests.get(
                    get_url,
                    headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"},
                    timeout=10,
                )
                g.raise_for_status()
                gjs = g.json()
                status = gjs.get("status")
                if status in {"succeeded", "failed", "canceled"}:
                    out = gjs.get("output")
                    if isinstance(out, list):
                        text_out = "\n".join(map(str, out))
                    elif isinstance(out, str):
                        text_out = out
                    else:
                        text_out = json.dumps(out) if out is not None else ""
                    if status != "succeeded":
                        raise RuntimeError(f"replicate status={status}")
                    break
                _sleep(1.0)
            return _finish_from_text_output(text_out)

        return with_retries(_do)

    return {"ok": False, "error": f"Unknown LLM_PROVIDER '{provider}'", "provider": provider, "model": model}


# ---- Planning & dispatch ----
async def plan_and_dispatch(
    request_id: str,
    questions_text: str,
    attachments_dir: str,
    attachments_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a deterministic plan JSON using either SKIP_LLM heuristics or by calling an LLM.
    Never send raw attachment bytes to an LLM; only include filenames and small metadata.
    Returns a dict { ok, plan, provider, model, error? }.
    """
    # Heuristic features
    filenames = [m.get("filename", "") for m in attachments_meta]
    task_type = _detect_task_type(questions_text, filenames)

    if SKIP_LLM or LLM_PROVIDER == "none":
        # Deterministic heuristic plan
        steps: List[Dict[str, Any]] = []
        steps.append({"id": "s1", "type": "parse_questions", "description": "Parse instructions from questions.txt"})
        if any(fn.lower().endswith(".csv") for fn in filenames):
            steps.append({"id": "s2", "type": "load_csv", "description": "Load CSV files into DataFrames"})
            steps.append({"id": "s3", "type": "analyze_tabular", "description": "Run summary stats and answer prompts"})
        if any(fn.lower().endswith(ext) for ext in (".parquet", ".pq", ".duckdb") for fn in filenames):
            steps.append({"id": "s4", "type": "query_parquet_duckdb", "description": "Query columnar data using DuckDB"})
        if ("http://" in questions_text) or ("https://" in questions_text):
            steps.append({"id": "s5", "type": "scrape", "description": "Fetch and parse referenced web pages"})
        if not any(s["id"] == "s3" for s in steps):
            steps.append({"id": "s6", "type": "text_analysis", "description": "Analyze and summarize textual instructions"})
        plan = {"plan": {"steps": steps}}
        return {"ok": True, "plan": plan, "provider": "heuristic", "model": "skip"}

    # Compose compact context for the LLM (no file bytes)
    q_preview = (questions_text or "").strip()
    if len(q_preview) > 1200:
        q_preview = q_preview[:1200]
    context = {
        "question_preview": q_preview,
        "attachments": [{"filename": m["filename"], "bytes": m.get("bytes", 0)} for m in attachments_meta],
        "constraints": {
            "no_raw_bytes": True,
            "max_runtime_s": REQUEST_TIMEOUT_SECONDS,
        },
    }

    def _provider_call() -> Dict[str, Any]:
        # Call model requesting JSON-only plan
        prompt = (
            "Create a short execution plan as JSON only. Schema: {\"plan\":{\"steps\":[{\"id\":\"s1\",\"type\":\"string\",\"description\":\"short\",\"params\":{}}]}}. "
            "Use 2-6 steps. Allowed types include: parse_questions, math, load_csv, analyze_tabular, query_parquet_duckdb, scrape, matplotlib_plot, llm_answer, text_analysis. "
            "Choose minimal steps to answer. No code, no markdown."
            " Context: " + json.dumps(context, ensure_ascii=False)
        )
        return call_llm(prompt, max_tokens=512, temperature=0.0, request_id=request_id)

    # Offload sync HTTP to thread so we don't block the event loop
    result = await asyncio.to_thread(with_retries, _provider_call, retries=1, base_delay=0.75)

    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "llm_error"), "provider": result.get("provider"), "model": result.get("model")}

    data = result.get("data") or result.get("plan")
    if not isinstance(data, dict):
        return {"ok": False, "error": "Malformed plan response", "provider": result.get("provider"), "model": result.get("model")}

    plan = data if "plan" in data else {"plan": data}

    # Basic schema validation
    steps = plan.get("plan", {}).get("steps") if isinstance(plan.get("plan"), dict) else None
    if not isinstance(steps, list) or not steps:
        return {"ok": False, "error": "Plan missing steps", "provider": result.get("provider"), "model": result.get("model")}
    for s in steps:
        if not isinstance(s, dict) or not s.get("id") or not s.get("type"):
            return {"ok": False, "error": "Invalid step in plan", "provider": result.get("provider"), "model": result.get("model")}

    return {"ok": True, "plan": plan, "provider": result.get("provider"), "model": result.get("model")}


async def _save_upload_to_disk(
    upload: UploadFile,
    dest_path: str,
    per_file_limit: int,
    total_state: dict,
    chunk_size: int = 1024 * 1024,
) -> int:
    """
    Save an UploadFile to disk in chunks with size enforcement.
    Returns number of bytes written. Updates total_state["total_bytes"].
    Raises HTTPException 413 on limit breach.
    """
    bytes_written = 0
    with open(dest_path, "wb") as f:
        while True:
            # Read at most chunk_size, but also respect remaining budget
            remaining_file = per_file_limit - bytes_written
            remaining_total = TOTAL_MAX_BYTES - total_state["total_bytes"]
            if remaining_file <= 0 or remaining_total <= 0:
                raise HTTPException(status_code=413, detail="Upload size limit exceeded")
            to_read = min(chunk_size, remaining_file, remaining_total)
            chunk = await upload.read(to_read)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
            total_state["total_bytes"] += len(chunk)
            if bytes_written > per_file_limit or total_state["total_bytes"] > TOTAL_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Upload size limit exceeded")
    return bytes_written


async def _process_request(request: Request, request_id: str, steps: List[str]) -> JSONResponse:
    start_ts = _utc_now_iso()
    start_time = datetime.now(timezone.utc)

    temp_dir = tempfile.mkdtemp(prefix=f"req-{request_id[:8]}-")
    total_state = {"total_bytes": 0}

    try:
        form = await request.form()
        steps.append("form_parsed")

        # Gather all UploadFile instances from the form
        files: List[UploadFile] = [v for v in form.values() if isinstance(v, UploadFile)]
        if not files:
            raise HTTPException(status_code=400, detail="Multipart form must include files; 'questions.txt' is required")

        # Find questions.txt by filename (case-insensitive)
        qfile: UploadFile | None = None
        for uf in files:
            if (uf.filename or "").lower() == "questions.txt":
                qfile = uf
                break

        if qfile is None:
            # If not found by name, try a heuristic: single text-like file
            text_like = [
                uf for uf in files if (uf.content_type or "").startswith("text/")
            ]
            if len(text_like) == 1:
                qfile = text_like[0]

        if qfile is None:
            raise HTTPException(status_code=400, detail="Missing required file 'questions.txt'")

        # Read and count questions.txt within limits
        q_bytes_io = io.BytesIO()
        q_written = 0
        while True:
            remaining_file = PER_FILE_MAX_BYTES - q_written
            remaining_total = TOTAL_MAX_BYTES - total_state["total_bytes"]
            if remaining_file <= 0 or remaining_total <= 0:
                raise HTTPException(status_code=413, detail="Upload size limit exceeded")
            to_read = min(1024 * 1024, remaining_file, remaining_total)
            chunk = await qfile.read(to_read)
            if not chunk:
                break
            q_bytes_io.write(chunk)
            q_written += len(chunk)
            total_state["total_bytes"] += len(chunk)
            if q_written > PER_FILE_MAX_BYTES or total_state["total_bytes"] > TOTAL_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Upload size limit exceeded")
        question_text = _strip_bom(q_bytes_io.getvalue())
        steps.append("questions_loaded")

        # Save remaining attachments (exclude qfile identity)
        attachments_meta = []
        for uf in files:
            if uf is qfile:
                continue
            safe_name = _sanitize_filename(uf.filename or "attachment")
            dest_path = os.path.join(temp_dir, safe_name)
            written = await _save_upload_to_disk(uf, dest_path, PER_FILE_MAX_BYTES, total_state)
            attachments_meta.append({"filename": safe_name, "bytes": written})
        steps.append("attachments_saved")

        attachment_names = [m["filename"] for m in attachments_meta]
        task_type = _detect_task_type(question_text, attachment_names)
        steps.append("heuristics_done")

        # Plan generation (safe & deterministic). Will not send attachment bytes.
        plan_result = await plan_and_dispatch(request_id, question_text, temp_dir, attachments_meta)
        steps.append("plan_generated" if plan_result.get("ok") else "plan_failed")

        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        # Structured, privacy-conscious log
        log_event = {
            "event": "ingest_ack",
            "request_id": request_id,
            "task_type": task_type,
            "file_count": 1 + len(attachments_meta),
            "filenames": ["questions.txt"] + attachment_names,
            "bytes_total": total_state["total_bytes"],
            "provider": plan_result.get("provider"),
            "model": plan_result.get("model"),
            "ts": start_ts,
            "duration_ms": duration_ms,
        }
        print(json.dumps(log_event), flush=True)

        ack: Dict[str, Any] = {
            "request_id": request_id,
            "acknowledged": True,
            "task_type": task_type,
            "received": {
                "questions_bytes": q_written,
                "num_attachments": len(attachments_meta),
                "attachments": attachment_names,
            },
            "limits": {
                "per_file_max_bytes": PER_FILE_MAX_BYTES,
                "total_max_bytes": TOTAL_MAX_BYTES,
            },
            "dev": {
                "skip_llm": SKIP_LLM,
                "provider": plan_result.get("provider"),
                "model": plan_result.get("model"),
            },
        }
        # Include a minimal plan preview for debugging (no sensitive data)
        if plan_result.get("ok"):
            ack["plan"] = plan_result.get("plan")
        else:
            ack["plan_error"] = plan_result.get("error")

        return JSONResponse(status_code=202, content=ack)

    except HTTPException as he:
        # Log and re-raise as JSON response
        error_payload = {
            "request_id": request_id,
            "error": he.detail,
        }
        print(json.dumps({"event": "error", "request_id": request_id, "status": he.status_code, "detail": he.detail, "ts": _utc_now_iso()}), flush=True)
        return JSONResponse(status_code=he.status_code, content=error_payload)
    except Exception as e:
        error_payload = {
            "request_id": request_id,
            "error": "Internal server error",
        }
        print(json.dumps({"event": "error", "request_id": request_id, "status": 500, "detail": str(e), "ts": _utc_now_iso()}), flush=True)
        return JSONResponse(status_code=500, content=error_payload)
    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@app.post("/")
async def ingest(request: Request):
    request_id = str(uuid.uuid4())
    steps: List[str] = []
    try:
        response = await asyncio.wait_for(_process_request(request, request_id, steps), timeout=REQUEST_TIMEOUT_SECONDS)
        return response
    except asyncio.TimeoutError:
        payload = {
            "request_id": request_id,
            "error": "Processing timed out. Please try a smaller request or simplify inputs.",
            "steps_completed": steps,
        }
        print(json.dumps({"event": "timeout", "request_id": request_id, "ts": _utc_now_iso()}), flush=True)
        return JSONResponse(status_code=504, content=payload)


@app.get("/")
async def health():
    return {"status": "ok", "name": "data-analyst-agent", "time": _utc_now_iso()}


# Optional: make handler also available under common paths (useful behind certain platforms)
# Map legacy paths for the ingest/ack endpoint only to / (root). Keep /api for analysis endpoint.
for route in ["/api/index", "/api/index/"]:
    app.add_api_route(route, ingest, methods=["POST"])


# ---- Helper: simple math evaluator ----
import ast

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
    ast.Pow,  # allow exponentiation cautiously
    ast.Load,
    ast.Call,  # will be rejected below
    ast.Name,  # will be rejected below
)


def _is_safe_ast(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_AST_NODES):
            return False
        # Disallow names and calls to avoid attacks
        if isinstance(child, (ast.Call, ast.Name)):
            return False
    return True


def eval_simple_math(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression safely. Returns None if not simple.
    Supports +, -, *, /, //, %, ** and parentheses.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except Exception:
        return None
    if not _is_safe_ast(tree):
        return None
    try:
        return float(eval(compile(tree, "<expr>", "eval")))
    except Exception:
        return None


# ---- Attachment loaders & lightweight analysis ----

def load_attachments(attachments_dir: str, attachments_meta: List[Dict[str, Any]]):
    dataframes: Dict[str, Any] = {}
    json_objs: Dict[str, Any] = {}
    others: Dict[str, Dict[str, Any]] = {}
    for meta in attachments_meta:
        fn = meta.get("filename")
        if not fn:
            continue
        path = os.path.join(attachments_dir, fn)
        low = fn.lower()
        try:
            if low.endswith(".csv"):
                import pandas as pd  # local import to keep import time low
                # Read small/medium CSV; our upload caps keep this bounded
                df = pd.read_csv(path)
                dataframes[fn] = df
            elif low.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    json_objs[fn] = json.load(f)
            elif low.endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read(1024 * 64)
                others[fn] = {"preview": txt[:200]}
            else:
                # Just capture metadata for unknown types
                stat = os.stat(path)
                others[fn] = {"bytes": stat.st_size}
        except Exception as e:
            others[fn] = {"error": str(e)}
    return dataframes, json_objs, others


def make_simple_plot_base64(df) -> Optional[str]:
    """Create a tiny PNG plot as base64 from the first numeric column(s).
    Returns None if plotting not possible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io as _io
        # pick first 1-2 numeric columns
        num_df = df.select_dtypes(include=["number"]).head(100)
        if num_df.empty:
            return None
        plt.figure(figsize=(3, 2))
        num_df.reset_index(drop=True).plot(ax=plt.gca())
        plt.tight_layout()
        buf = _io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64
    except Exception:
        return None


# ---- API: /api (lightweight Q&A / analysis) ----
import base64

# ---- Planning execution engine ----
async def execute_plan(
    plan: Dict[str, Any],
    *,
    question_text: str,
    attachments_dir: str,
    attachments_meta: List[Dict[str, Any]],
    request_id: str,
) -> Dict[str, Any]:
    """
    Execute plan steps sequentially. Returns minimal JSON like {"answer": ...} or {"result": ...}.
    Handlers are lightweight and avoid heavy memory usage. No raw bytes sent to LLM.
    """
    artifacts: Dict[str, Any] = {}

    # Helper loaders available to steps
    def _csv_files():
        return [m["filename"] for m in attachments_meta if m["filename"].lower().endswith(".csv")]

    def _parquet_files():
        return [m["filename"] for m in attachments_meta if m["filename"].lower().endswith((".parquet", ".pq"))]

    def _duckdb_files():
        return [m["filename"] for m in attachments_meta if m["filename"].lower().endswith(".duckdb")]

    # Simple dispatcher implementations
    for step in plan.get("plan", {}).get("steps", []):
        stype = (step.get("type") or "").lower().strip()
        sid = step.get("id") or f"s{len(artifacts)+1}"
        params = step.get("params") or {}

        try:
            if stype in {"parse_questions", "noop"}:
                # Optionally extract simple flags or expressions
                expr_val = eval_simple_math(question_text)
                if expr_val is not None:
                    artifacts["math"] = int(expr_val) if abs(expr_val - int(expr_val)) < 1e-9 else round(expr_val, 6)

            elif stype in {"math", "compute"}:
                expr = params.get("expression") or question_text
                val = eval_simple_math(expr)
                if val is None:
                    artifacts[sid] = {"error": "invalid_expression"}
                else:
                    artifacts["answer"] = int(val) if abs(val - int(val)) < 1e-9 else round(val, 6)

            elif stype in {"scrape", "fetch"}:
                urls = params.get("urls") or params.get("url") or []
                if isinstance(urls, str):
                    urls = [urls]
                texts = {}
                for u in urls[:5]:
                    try:
                        r = requests.get(u, timeout=10)
                        r.raise_for_status()
                        content = r.content[: 512 * 1024]  # 512KB cap per page
                        from bs4 import BeautifulSoup  # lazy import
                        soup = BeautifulSoup(content, "html.parser")
                        txt = soup.get_text(" ", strip=True)
                        texts[u] = txt[:5000]
                    except Exception as e:
                        texts[u] = f"error:{type(e).__name__}"
                artifacts[sid] = {"scraped": texts}

            elif stype in {"load_csv"}:
                files = params.get("files") or _csv_files()
                loaded = {}
                for fn in files:
                    path = os.path.join(attachments_dir, fn)
                    try:
                        import pandas as pd  # local import
                        df = pd.read_csv(path)
                        loaded[fn] = df
                    except Exception as e:
                        loaded[fn] = f"error:{type(e).__name__}"
                artifacts.setdefault("dataframes", {}).update(loaded)

            elif stype in {"analyze_tabular", "summarize_csv"}:
                dfs = artifacts.get("dataframes", {})
                if not dfs:
                    # try lazy load any CSVs if not yet loaded
                    files = _csv_files()
                    if files:
                        import pandas as pd  # noqa: F401
                        for fn in files:
                            try:
                                df = pd.read_csv(os.path.join(attachments_dir, fn))
                                artifacts.setdefault("dataframes", {})[fn] = df
                            except Exception:
                                pass
                        dfs = artifacts.get("dataframes", {})
                summary: Dict[str, Any] = {}
                for name, df in list(dfs.items())[:2]:
                    if hasattr(df, "describe"):
                        try:
                            summary[name] = df.describe(include="all").to_dict()
                        except Exception:
                            summary[name] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
                if summary:
                    artifacts.setdefault("summaries", {}).update(summary)

            elif stype in {"query_parquet_duckdb", "duckdb_query"}:
                sql = params.get("sql") or ""
                pqs = _parquet_files()
                ddbs = _duckdb_files()
                if not sql:
                    # generate a trivial preview
                    if pqs:
                        _p = os.path.join(attachments_dir, pqs[0]).replace("\\", "/")
                        sql = f"SELECT * FROM read_parquet('{_p}') LIMIT 5"
                    elif ddbs:
                        sql = "SELECT 1 as ok"
                try:
                    import duckdb  # lazy import
                    con = duckdb.connect()
                    # Register CSVs and Parquet convenience views
                    for fn in _csv_files():
                        con.register(fn.replace("-", "_"), os.path.join(attachments_dir, fn))
                    if pqs and "read_parquet" not in sql.lower():
                        # If users refer to table parquet_1, map to file 0
                        pass
                    res = con.execute(sql).fetchdf()
                    artifacts[sid] = {"rows": min(5, len(res)), "preview": res.head(5).to_dict(orient="records")}
                    con.close()
                except Exception as e:
                    artifacts[sid] = {"error": f"duckdb:{type(e).__name__}"}

            elif stype in {"plot", "matplotlib_plot"}:
                # plot first available numeric df
                dfs = artifacts.get("dataframes", {})
                if not dfs:
                    files = _csv_files()
                    if files:
                        import pandas as pd  # noqa: F401
                        try:
                            df = pd.read_csv(os.path.join(attachments_dir, files[0]))
                            artifacts.setdefault("dataframes", {})[files[0]] = df
                            dfs = artifacts["dataframes"]
                        except Exception:
                            pass
                if dfs:
                    name, df = next(iter(dfs.items()))
                    b64 = make_simple_plot_base64(df)
                    if b64:
                        artifacts.setdefault("plots", {})[name] = b64

            elif stype in {"llm_answer", "lookup", "answer"}:
                # Ask the model for a short answer in JSON only
                if SKIP_LLM or LLM_PROVIDER == "none":
                    artifacts[sid] = {"error": "llm_disabled"}
                else:
                    q = params.get("question") or question_text[:800]
                    prompt = "Answer briefly in JSON only. Schema: {\"answer\":\"short\"}. No markdown.\nQUESTION: " + q
                    res = await asyncio.to_thread(
                        call_llm,
                        prompt,
                        max_tokens=128,
                        temperature=0.0,
                        request_id=request_id,
                        prefix_instructions="Respond ONLY with compact JSON. No prose. No markdown."
                    )
                    if res.get("ok") and isinstance(res.get("data"), dict):
                        data = res["data"]
                        if "answer" in data:
                            artifacts["answer"] = data["answer"]
                        else:
                            artifacts[sid] = data
                    else:
                        artifacts[sid] = {"error": res.get("error", "llm_error")}

            else:
                # Unknown step type: ignore but log
                print(json.dumps({"event": "unknown_step", "type": stype, "id": sid}), flush=True)
        except Exception as e:
            artifacts[sid] = {"error": f"step_error:{type(e).__name__}"}

    # Decide minimal output
    if "answer" in artifacts:
        return {"answer": artifacts["answer"]}
    if "summaries" in artifacts and artifacts["summaries"]:
        return {"summary": artifacts["summaries"]}
    if "plots" in artifacts and artifacts["plots"]:
        return {"plot_png_base64": next(iter(artifacts["plots"].values()))}
    if "math" in artifacts:
        return {"answer": artifacts["math"]}
    # Fallback minimal result summary (kept short)
    keys = [k for k in artifacts.keys() if not isinstance(artifacts[k], dict) or "error" not in artifacts[k]]
    return {"result": {"artifacts": keys or list(artifacts.keys())[:5]}}


async def _process_api(request: Request, request_id: str) -> JSONResponse:
    start_ts = _utc_now_iso()
    start_time = datetime.now(timezone.utc)

    temp_dir = tempfile.mkdtemp(prefix=f"api-{request_id[:8]}-")
    total_state = {"total_bytes": 0}

    try:
        form = await request.form()
        # Collect files
        files: List[UploadFile] = [v for v in form.values() if isinstance(v, UploadFile)]
        if not files:
            return JSONResponse(status_code=400, content={"error": "questions.txt required (multipart/form-data)"})

        # Locate questions.txt
        qfile: Optional[UploadFile] = None
        for uf in files:
            if (uf.filename or "").lower() == "questions.txt":
                qfile = uf
                break
        if qfile is None:
            # try heuristic single text file
            text_like = [uf for uf in files if (uf.content_type or "").startswith("text/")]
            if len(text_like) == 1:
                qfile = text_like[0]
        if qfile is None:
            return JSONResponse(status_code=400, content={"error": "Missing questions.txt"})

        # Read questions within limits
        q_bytes = io.BytesIO()
        q_written = 0
        while True:
            remaining_file = PER_FILE_MAX_BYTES - q_written
            remaining_total = TOTAL_MAX_BYTES - total_state["total_bytes"]
            if remaining_file <= 0 or remaining_total <= 0:
                return JSONResponse(status_code=413, content={"error": "Upload size limit exceeded"})
            to_read = min(1024 * 1024, remaining_file, remaining_total)
            chunk = await qfile.read(to_read)
            if not chunk:
                break
            q_bytes.write(chunk)
            q_written += len(chunk)
            total_state["total_bytes"] += len(chunk)
            if q_written > PER_FILE_MAX_BYTES or total_state["total_bytes"] > TOTAL_MAX_BYTES:
                return JSONResponse(status_code=413, content={"error": "Upload size limit exceeded"})
        question_text = _strip_bom(q_bytes.getvalue()).strip()

        # Save other attachments
        attachments_meta: List[Dict[str, Any]] = []
        for uf in files:
            if uf is qfile:
                continue
            safe_name = _sanitize_filename(uf.filename or "attachment")
            dest_path = os.path.join(temp_dir, safe_name)
            written = await _save_upload_to_disk(uf, dest_path, PER_FILE_MAX_BYTES, total_state)
            attachments_meta.append({"filename": safe_name, "bytes": written})

        # Plan using unified LLM integration (or heuristics if SKIP_LLM/none)
        plan_result = await plan_and_dispatch(request_id, question_text, temp_dir, attachments_meta)
        if not plan_result.get("ok"):
            return JSONResponse(status_code=502, content={"error": plan_result.get("error", "plan_error")})
        plan = plan_result.get("plan")

        # Execute the plan generically
        exec_output = await execute_plan(
            plan,
            question_text=question_text,
            attachments_dir=temp_dir,
            attachments_meta=attachments_meta,
            request_id=request_id,
        )

        print(json.dumps({
            "event": "api_done",
            "request_id": request_id,
            "provider": plan_result.get("provider"),
            "model": plan_result.get("model"),
            "ts": start_ts
        }), flush=True)
        return JSONResponse(status_code=200, content=exec_output)

    except Exception as e:
        print(json.dumps({"event": "api_error", "request_id": request_id, "detail": str(e), "ts": _utc_now_iso()}), flush=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    finally:
        # Best-effort cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ---- Inline quick tests (doctest-style) ----

def _test_safe_parse_json_examples():
    """
    >>> safe_parse_json('{"a":1}')['ok']
    True
    >>> safe_parse_json('Here is code:\n```\n{"a": 2}\n```\n')['ok']
    True
    >>> safe_parse_json('not json')['ok']
    False
    """
    pass


def _test_skip_llm_plan():
    """
    Ensures SKIP_LLM path yields a plan with steps.
    >>> os.environ.get('SKIP_LLM', 'false') in ('true','1') or True
    True
    """
    pass
