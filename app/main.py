from pathlib import Path
from time import perf_counter
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, HttpUrl
import requests
from app.config import RAW_DATA_PATH
from app.ingest import ingest_directory, ingest_path
from app.pipeline import answer
from app.store import get_store

app = FastAPI(title="Frontier RAG", version="0.1.0",
              description="ChromaDB + BM25 hybrid RAG with Groq generation and retrieval experiments.")

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|hyde|graph|agentic)$")
    top_k: int = Field(default=5, ge=1, le=12)
    latency_budget_ms: int = Field(default=2500, ge=100, le=120000)

class URLRequest(BaseModel):
    url: HttpUrl

@app.get("/health")
def health():
    try: count = get_store().count()
    except Exception: count = 0
    return {"status":"ok", "indexed_chunks":count}

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/ingest")
def ingest():
    try: return ingest_directory()
    except Exception as exc: raise HTTPException(500, f"Ingestion failed: {type(exc).__name__}: {exc}")

@app.post("/api/ingest/url")
def ingest_url(body: URLRequest):
    url = str(body.url)
    # Basic SSRF guard: allow public HTTP(S), reject local/private IP literal targets.
    import ipaddress, socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in {"http","https"}: raise HTTPException(400, "Only HTTP(S) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
        if any(ipaddress.ip_address(x[4][0]).is_private or ipaddress.ip_address(x[4][0]).is_loopback or ipaddress.ip_address(x[4][0]).is_link_local for x in infos):
            raise HTTPException(400, "Private/local hosts are not allowed")
    except socket.gaierror: raise HTTPException(400, "Hostname could not be resolved")
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent":"FrontierRAG/1.0"})
        resp.raise_for_status()
        if len(resp.content) > 15_000_000: raise HTTPException(413, "Source exceeds 15 MB limit")
        name = Path(parsed.path).name or "download.txt"
        if not Path(name).suffix: name += ".txt"
        RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        dest = RAW_DATA_PATH / Path(name).name
        dest.write_bytes(resp.content)
        return ingest_directory()
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400, f"URL ingestion failed: {exc}")

@app.post("/api/query")
def query(body: QueryRequest):
    try: return answer(body.question, body.mode, body.top_k, body.latency_budget_ms)
    except (RuntimeError, ValueError) as exc: raise HTTPException(400, str(exc))
    except Exception as exc: raise HTTPException(500, f"Query failed: {type(exc).__name__}: {exc}")

@app.post("/api/evaluate")
def evaluate():
    try:
        from app.evaluate import run_evaluation
        return run_evaluation()
    except Exception as exc: raise HTTPException(500, f"Evaluation failed: {exc}")
