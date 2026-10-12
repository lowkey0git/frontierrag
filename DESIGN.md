# Design decisions and benchmark protocol

## Request path
FastAPI validation → query strategy → Chroma dense search + BM25 → weighted reciprocal-rank fusion → optional cross-encoder reranking → sentence-level context selection → Groq completion → answer with source IDs and source records.

## Data path
Public URL or local file → extraction → paragraph normalization → embedding-similarity chunk boundary checks + hard character limit → normalized sentence-transformer vectors → persistent Chroma upsert + metadata → lightweight entity/chunk adjacency file → invalidate BM25 cache.

## Decisions and limitations
- **ChromaDB local persistence:** simple ownership, inspectability, and local development. For large multi-instance deployments, use managed/remote vector infrastructure or a separately operated Chroma server; local disk is not shared safely across arbitrary replicas.
- **BM25 + dense fusion:** complementary lexical exact-match and semantic retrieval. RRF avoids directly mixing incomparable score scales. `HYBRID_ALPHA` changes dense-vs-lexical rank contribution.
- **Embedding model:** `all-MiniLM-L6-v2` is a fast baseline. Benchmark domain-specific/multilingual alternatives against your actual corpus; re-embed and rebuild index when model changes.
- **Semantic chunking:** similarity threshold and character bounds are heuristic; tune using retrieval labels. Production ingestion should add language-aware segmentation, table handling, OCR, deduplication, versioning, and retry/dead-letter behavior.
- **Reranking:** optional due to model download and inference cost. Enable `ENABLE_RERANK=true` only after measuring relevance gains and p95 latency.
- **HyDE:** extra Groq generation; may raise recall for underspecified queries but risks query drift.
- **Graph RAG:** proper-noun co-occurrence graph is intentionally an experiment. It is not robust entity resolution and can connect unrelated names. Replace with NER, canonical entity IDs, typed edges, and provenance before relying on it.
- **Agentic decomposition:** decomposes into up to four subquestions, retrieves each, deduplicates and synthesizes. More coverage, but extra model calls and token cost.
- **Grounding:** answer prompt demands abstention and source IDs. Prompting alone cannot guarantee zero hallucinations. Evaluate citation support and abstention; for sensitive domains add deterministic citation validation and human review.
- **Latency budget:** measured wall-clock time is returned and compared to the user's budget. It is not a hard deadline; LLM, model warmup, cold starts, and network variability affect latency.

## Evaluation protocol
1. Build a domain-representative set of at least 100-300 queries, including unanswerable questions and multi-hop questions.
2. Label relevant source/chunk IDs independently.
3. Compare `dense`, `hybrid`, `hyde`, `graph`, `agentic`, and hybrid+rerank.
4. Report Recall@5/10, MRR, nDCG@k, citation precision/recall, answer correctness, abstention accuracy, p50/p95 end-to-end latency, tokens/cost, and failure rate.
5. Run each mode repeatedly after warm-up. Segment results by simple vs multi-hop and by latency budget.
6. Select the lowest-cost mode that satisfies your quality target and p95 budget; route difficult queries to slower modes only if the measured benefit justifies it.

The included evaluator is a starter smoke test. Its source-level labels and small sample do not establish system quality. Extend it to chunk-level ground truth and human-rated answer/citation quality before making production architecture claims.
