from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", str(DATA_DIR / "raw")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "12"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1400"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "350"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.55"))
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
ENABLE_GRAPH = os.getenv("ENABLE_GRAPH", "true").lower() == "true"
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
