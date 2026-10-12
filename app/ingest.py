from pathlib import Path
import hashlib, re
from bs4 import BeautifulSoup
from pypdf import PdfReader
from app.config import RAW_DATA_PATH, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS
from app.store import get_store
from app.retrieval import get_embedder, invalidate_bm25
from app.graph import update_graph

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    return raw

def split_paragraphs(text: str):
    text = text.replace("\r\n", "\n")
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def semantic_chunks(text: str):
    """Group adjacent paragraphs until semantic similarity drops or size cap is reached."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    embedder = get_embedder()
    chunks, current = [], [paragraphs[0]]
    current_len = len(paragraphs[0])
    for para in paragraphs[1:]:
        para_len = len(para)
        too_large = current_len + para_len + 1 > CHUNK_MAX_CHARS
        should_test = current_len >= CHUNK_MIN_CHARS
        split_semantic = False
        if should_test and not too_large:
            import numpy as np
            vecs = embedder.encode([" ".join(current[-2:]), para], normalize_embeddings=True)
            split_semantic = float(np.dot(vecs[0], vecs[1])) < 0.42
        if too_large or split_semantic:
            chunks.append("\n\n".join(current))
            current, current_len = [para], para_len
        else:
            current.append(para)
            current_len += para_len + 1
    if current:
        chunks.append("\n\n".join(current))
    # Hard split exceptionally long paragraphs; prevents oversized prompts.
    output = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_MAX_CHARS:
            output.append(chunk)
        else:
            output.extend(chunk[i:i+CHUNK_MAX_CHARS] for i in range(0, len(chunk), CHUNK_MAX_CHARS))
    return output

def ingest_path(path: Path):
    text = extract_text(path)
    chunks = semantic_chunks(text)
    if not chunks:
        return {"file": path.name, "chunks_added": 0, "status": "empty"}
    source_id = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.sha1(f"{source_id}:{i}:{chunk}".encode()).hexdigest()
        ids.append(chunk_id); docs.append(chunk)
        metas.append({"source": path.name, "source_id": source_id, "chunk_index": i, "char_count": len(chunk)})
    # Chroma upsert makes repeat ingestion idempotent for identical content/IDs.
    embeddings = get_embedder().encode(docs, normalize_embeddings=True).tolist()
    collection = get_store()
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    update_graph(ids, docs, metas)
    invalidate_bm25()
    return {"file": path.name, "chunks_added": len(chunks), "status": "ok"}

def ingest_directory():
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    supported = {".txt", ".md", ".html", ".htm", ".pdf"}
    results = [ingest_path(p) for p in sorted(RAW_DATA_PATH.rglob("*")) if p.is_file() and p.suffix.lower() in supported]
    return {"files_processed": len(results), "chunks_added": sum(x["chunks_added"] for x in results), "results": results}
