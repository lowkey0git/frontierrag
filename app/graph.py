"""Lightweight entity co-occurrence graph baseline for graph-RAG experiments."""
import re, json
from pathlib import Path
GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "graph.json"
_graph = None
def _load():
    global _graph
    if _graph is None:
        try: _graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        except Exception: _graph = {"entities": {}}
    return _graph

def entities(text):
    # Heuristic proper-noun phrases; replace with domain NER/entity linking for serious use.
    found = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    stop = {"The", "This", "That", "Chapter", "Mr", "Mrs", "Dr"}
    return sorted(set(x for x in found if x not in stop and len(x) > 2))

def update_graph(ids, docs, metas):
    g = _load()
    for chunk_id, doc, meta in zip(ids, docs, metas):
        for entity in entities(doc):
            node = g["entities"].setdefault(entity, [])
            if chunk_id not in node: node.append(chunk_id)
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(g), encoding="utf-8")

def expand_graph(rows, limit=3):
    g = _load(); by_id = {}
    for row in rows: by_id[row["id"]] = row
    for row in list(rows):
        for entity, ids in g.get("entities", {}).items():
            if row["id"] in ids:
                for linked in ids[:limit+1]:
                    if linked not in by_id:
                        by_id[linked] = {"id": linked, "text": "", "metadata": {"graph_entity": entity}, "score": 0.0}
    # Fill linked chunks from Chroma in one query.
    missing = [k for k,v in by_id.items() if not v["text"]]
    if missing:
        from app.store import get_store
        got = get_store().get(ids=missing, include=["documents", "metadatas"])
        for i,d,m in zip(got["ids"], got["documents"], got["metadatas"]):
            by_id[i].update({"text": d or "", "metadata": m or {}, "graph_expanded": True})
    return list(by_id.values())
