from functools import lru_cache
import re, threading
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, CANDIDATE_K, HYBRID_ALPHA, ENABLE_RERANK, RERANK_MODEL
from app.store import get_store

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

_bm25_lock = threading.Lock()
_bm25_cache = None
_reranker = None

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9_'-]+\b", text.lower())

def invalidate_bm25():
    global _bm25_cache
    with _bm25_lock:
        _bm25_cache = None

def _all_records():
    col = get_store()
    count = col.count()
    if not count:
        return []
    result = col.get(include=["documents", "metadatas"])
    return [{"id": i, "text": d or "", "metadata": m or {}} for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])]

def _get_bm25():
    global _bm25_cache
    with _bm25_lock:
        if _bm25_cache is None:
            records = _all_records()
            _bm25_cache = (records, BM25Okapi([tokenize(r["text"]) for r in records]) if records else None)
        return _bm25_cache

def dense_search(query, k=CANDIDATE_K, query_vector=None):
    col = get_store()
    if col.count() == 0: return []
    vec = query_vector if query_vector is not None else get_embedder().encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[vec], n_results=min(k, col.count()), include=["documents", "metadatas", "distances"])
    out = []
    for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"id": i, "text": doc, "metadata": meta, "dense_score": 1.0-float(dist), "score": 1.0-float(dist)})
    return out

def bm25_search(query, k=CANDIDATE_K):
    records, model = _get_bm25()
    if not model: return []
    scores = model.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1][:k]
    return [{"id": records[i]["id"], "text": records[i]["text"], "metadata": records[i]["metadata"], "bm25_score": float(scores[i]), "score": float(scores[i])} for i in idx if scores[i] > 0]

def hybrid_search(query, k=CANDIDATE_K, alpha=HYBRID_ALPHA, query_vector=None):
    dense = dense_search(query, k, query_vector)
    lexical = bm25_search(query, k)
    # Weighted reciprocal-rank fusion is robust to incomparable raw score scales.
    fused = {}
    for weight, rows in ((alpha, dense), (1-alpha, lexical)):
        for rank, row in enumerate(rows, 1):
            item = fused.setdefault(row["id"], {**row, "rrf": 0.0})
            item["rrf"] += weight / (60 + rank)
            if row.get("dense_score") is not None: item["dense_score"] = row["dense_score"]
            if row.get("bm25_score") is not None: item["bm25_score"] = row["bm25_score"]
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:k]

def rerank(query, rows, top_k):
    global _reranker
    if not ENABLE_RERANK or not rows: return rows[:top_k]
    try:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL)
        scores = _reranker.predict([(query, r["text"]) for r in rows])
        return [dict(r, rerank_score=float(s)) for r, s in sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[:top_k]]
    except Exception:
        # Retrieval remains available if optional reranker model cannot initialize.
        return rows[:top_k]

def compress_context(query, rows, max_chars=9000):
    """Extract query-relevant sentences while retaining each original chunk/source."""
    terms = set(tokenize(query))
    selected, used = [], 0
    for row in rows:
        sentences = re.split(r"(?<=[.!?])\s+", row["text"])
        ranked = sorted(sentences, key=lambda s: sum(t in set(tokenize(s)) for t in terms), reverse=True)
        picked, n = [], 0
        for sentence in ranked:
            if not sentence.strip(): continue
            if n + len(sentence) > 1800: continue
            picked.append(sentence); n += len(sentence)
            if len(picked) >= 5: break
        snippet = " ".join(picked) if picked else row["text"][:1800]
        snippet = snippet[:max_chars-used]
        if not snippet: break
        selected.append({**row, "context_text": snippet})
        used += len(snippet)
        if used >= max_chars: break
    return selected
