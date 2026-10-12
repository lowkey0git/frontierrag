# Frontier RAG — production-oriented reference implementation

A practical RAG service built around FastAPI, ChromaDB, BM25, sentence-transformer embeddings, optional cross-encoder reranking, and the live Groq API.

## What this is
A deployable **reference implementation**, not a claim of production validation. It includes:
- Public-domain text download (Project Gutenberg) and local-file ingestion (`.txt`, `.md`, `.html`, `.pdf`).
- Persistent Chroma vector index with document metadata and stable chunk IDs.
- Lexical BM25 + dense vector retrieval; configurable weighted reciprocal-rank fusion.
- Semantic-boundary chunking using embedding similarity, with size limits.
- Optional cross-encoder reranking and context compression.
- Groq-backed answer generation, query decomposition, and HyDE.
- Experimental graph expansion and adaptive/agentic retrieval modes.
- Evaluation harness that records retrieval/answer latency, Recall@k, MRR, and answer-grounding checks against a small editable test set.
- FastAPI endpoints, a small browser UI, Dockerfile, and Render deployment manifest.

## Architecture / trade-offs
1. **Baseline (default):** query → dense + BM25 → reciprocal-rank fusion → optional rerank → bounded context → Groq answer with source citations. Best starting point for a latency-conscious general corpus.
2. **HyDE:** asks the LLM to draft a hypothetical answer/document, embeds it, and retrieves against that vector. May improve semantic recall for vague queries, but adds one LLM round-trip and can amplify incorrect assumptions.
3. **Graph expansion:** builds lightweight entity co-occurrence links during ingestion and expands retrieved context to linked chunks. Useful for cross-document entity relationships; graph quality depends on extraction and can add noise. This is a lightweight graph-RAG baseline, not a knowledge graph with a domain ontology.
4. **Agentic/adaptive:** decomposes multi-part questions and conditionally broadens retrieval when evidence looks weak. Can improve coverage but adds LLM calls and latency; use only when the latency budget permits.
5. **Reranking:** a cross-encoder improves ordering after broad retrieval but adds CPU/GPU cost. Keep disabled by default for low-latency deployment.
6. **Compression:** sentence-level evidence selection reduces prompt size, but aggressive compression can remove qualifications. Original chunk citations are retained.

No architecture wins universally. Benchmark on your own representative questions and corpus before selecting a mode.

## Quick start (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY in .env
python scripts/download_public_data.py
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`. API docs: `http://127.0.0.1:8000/docs`.

The initial data downloader retrieves Project Gutenberg public-domain works (including *Frankenstein* and *The Adventures of Sherlock Holmes*). Verify the applicable copyright status for your jurisdiction and any alternate sources you add. The raw downloaded files are saved in `data/raw/`, so you own and can inspect the local copies; download URLs and attribution are stored in `data/manifest.json`.

## API
- `GET /health`
- `POST /api/ingest` — ingest files already placed under `data/raw/`
- `POST /api/ingest/url` — download a single text URL (only use sources you have rights to ingest)
- `POST /api/query` — `{"question":"...", "mode":"hybrid", "top_k":5, "latency_budget_ms":2500}`
- `POST /api/evaluate` — run the included benchmark questions and return metrics

Modes: `hybrid`, `dense`, `hyde`, `graph`, `agentic`. `hybrid` is the recommended baseline.

## Configuration
All secrets belong in `.env`; never commit it. See `.env.example`. `GROQ_API_KEY` is required for generated answers and advanced LLM-driven modes. Embedding and retrieval can be used without Groq, but `/api/query` returns a clear configuration error when generation is requested without a key.

## Benchmarking
```bash
python -m app.evaluate --questions data/eval_questions.json
```
Results are written to `data/eval_results.json`. Run multiple trials and compare p50/p95 latency, Recall@k, MRR, citation validity, and answer quality on a human-reviewed test set. The included set is only a smoke test; it is not statistically representative. The `latency_budget_ms` is an architecture-selection hint, not a hard real-time guarantee.

## Docker
```bash
docker build -t frontier-rag .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" frontier-rag
```

## Deploy to Render
1. Push this repository to your own GitHub repository.
2. In Render, create a Web Service from that repository. The included `render.yaml` is a starting point.
3. Add `GROQ_API_KEY` as a secret environment variable in the hosting dashboard.
4. Attach persistent storage at `/app/data` if your plan supports it. Without persistent storage, local Chroma data may be lost on redeploy.
5. Deploy, then test `/health`, `/docs`, and the UI.

Do not expose this demo publicly with unrestricted ingestion enabled. Add authentication, rate limiting, tenant isolation, quotas, and content-security controls before serving untrusted users at scale. The default API is a starter service, not a hardened multi-tenant SaaS.

## Push to your GitHub
Create an empty repository on GitHub, then from this directory:
```bash
git init
git add .
git commit -m "Build Frontier RAG pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
Replace the placeholders. Do not commit `.env`, API keys, private corpora, or benchmark data containing sensitive information.
