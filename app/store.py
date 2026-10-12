import chromadb
from app.config import CHROMA_PATH

_client = None
_collection = None

def get_store():
    global _client, _collection
    if _collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name="frontier_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection
