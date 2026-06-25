"""
rag_pipeline.py
===============
Wires together the full RAG pipeline:
  HybridRetriever → RetrievalVerifier → AnswerFusion

Flow
----
User query
    │
    ▼
[query_guard]          ← block unsafe queries before any retrieval
    │
    ▼
[HybridRetriever]      ← dense (BGE-M3) + BM25 + RRF merge
    │
    ▼
[verify_retrieval]     ← check if results are good enough
    │
    ├── not_found  → safe fallback message
    │
    └── confident / borderline
            │
            ▼
        [fuse_chunks]  ← deduplicate + combine + cite
            │
            ▼
        Final answer

Usage
-----
    from rag_pipeline import RAGPipeline
    p = RAGPipeline()
    print(p.answer("What are symptoms of malaria?", verbose=True))
"""

from hybrid_retrieval   import HybridRetriever
from retrieval_verifier import verify_retrieval
from answer_fusion      import fuse_chunks
from query_guard        import check_query_policy


# ── Settings ──────────────────────────────────────────────────────────────────

CHUNKS_FILE = "../all_chunks.jsonl"
FINAL_TOP_K = 5


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever(chunks_file=CHUNKS_FILE)

    def answer(self, query: str, verbose: bool = False) -> str:
        """
        Answer a user query end-to-end.

        Args:
            query   : user question string
            verbose : print retrieval details for debugging

        Returns:
            Final answer string or safe fallback.
        """
        # Step 0: Block unsafe queries before any retrieval
        guard = check_query_policy(query)
        if guard.blocked:
            return guard.message

        # Step 1: Hybrid retrieval — dense + BM25 + RRF
        results = self.retriever.retrieve(
            query   = query,
            top_k   = FINAL_TOP_K,
            verbose = verbose,
        )

        # Step 2: Convert to verifier format
        # RRF scores range 0.01–0.033; normalise to 0–1 for threshold checks
        # max RRF = 2/61 ≈ 0.0328 (rank=1 in both lists with k=60)
        MAX_RRF = 2.0 / (60 + 1)
        chunks    = []
        distances = []
        for r in results:
            chunks.append({
                "text":          r.text,
                "chunk_type":    r.chunk_type,
                "section_title": r.section_title,
                "metadata":      r.metadata,
            })
            normalised_sim = min(r.score / MAX_RRF, 1.0)
            distances.append(1.0 - normalised_sim)

        # Step 3: Verify retrieval quality
        verification = verify_retrieval(query, chunks, distances)

        if verbose:
            print(f"\n[Verifier] status={verification.status}  "
                  f"top_sim={verification.top_similarity:.3f}")

        # Step 4: Return fallback or fused answer
        if verification.status == "not_found":
            return verification.fallback_message

        return fuse_chunks(
            chunks           = verification.valid_chunks[:3],
            query            = query,
            borderline_prefix= verification.fallback_message or "",
        )