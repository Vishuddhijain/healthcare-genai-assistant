"""
hybrid_retrieval.py
===================
Priority 3: Combines Dense (BGE-M3) + BM25 using Reciprocal Rank Fusion (RRF).

This module owns retrieval ranking only. It keeps the existing retrieval
architecture intact:
  1. route query by category
  2. dense retrieval
  3. BM25 retrieval
  4. RRF merge
  5. metadata-aware adjustment
  6. cross-encoder reranking

The public return type remains list[RetrievalResult], so rag_pipeline.py,
retrieval_verifier.py, and answer_fusion.py do not need changes.
"""

import os
import re
from typing import Optional

from retrieval import DenseRetriever, RetrievalResult, TOP_K
from bm25_retrieval import BM25Index


# -- RRF Settings -------------------------------------------------------------

RRF_K = 60       # standard constant; do not change without benchmarking
FINAL_TOP_K = 5  # how many chunks to pass to retrieval_verifier after reranking
RERANK_CANDIDATE_K = 20
MAX_RRF_SCORE = 2.0 / (RRF_K + 1)

# Configurable reranker. Use ENABLE_RERANKER=0 for fast debugging, or set
# RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 to swap models.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1").lower() not in {"0", "false", "no"}

# Blend applied after RRF. Final RetrievalResult.score is rescaled back into
# the original RRF range so the verifier thresholds keep working unchanged.
RERANKER_WEIGHT = 0.70
RRF_WEIGHT = 0.30
METADATA_BOOST_WEIGHT = 0.25
METADATA_PENALTY_WEIGHT = 0.18


# -- Category constants -------------------------------------------------------

CAT_SCHEME = "Government Healthcare Scheme"
CAT_DISEASE = "Disease & Awareness"


# -- Query Router -------------------------------------------------------------

_SCHEME_KEYWORDS = {
    "scheme", "yojana", "karyakram",
    "jssk", "jsy", "pmmvy", "pmjay", "pm-jay", "abha", "nhm",
    "ayushman", "nikshay", "himcare", "suman", "pmsby", "pmjjby",
    "apply", "application", "eligible", "eligibility", "enrol", "enroll",
    "register", "registration", "beneficiary", "beneficiaries",
    "benefit", "benefits", "cash", "rupees", "incentive",
    "reimbursement", "subsidy",
    "documents required", "documents needed",
    "govt scheme", "government scheme", "health scheme",
    "bpl", "apl", "sc/st",
}

_CROSS_CATEGORY_KEYWORDS = {
    "financial support", "financial assistance", "government support",
    "free treatment", "free medicine", "free drugs", "free diagnosis",
    "government hospital", "government health",
}


def _route_query(query: str) -> Optional[str]:
    """
    Classify query into a ChromaDB category for targeted retrieval.

    Returns:
        CAT_SCHEME  -> only government scheme chunks
        CAT_DISEASE -> only disease/awareness chunks
        None        -> search all chunks for cross-category queries
    """
    q = query.lower()

    for keyword in _CROSS_CATEGORY_KEYWORDS:
        if keyword in q:
            return None

    for keyword in _SCHEME_KEYWORDS:
        if keyword in q:
            return CAT_SCHEME

    return CAT_DISEASE


# -- Metadata-Aware Filtering -------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "this", "that", "these", "those", "it", "its", "as", "if",
    "so", "not", "also", "may", "should", "into", "than", "more", "about",
    "what", "which", "when", "where", "who", "whom", "how", "why",
}

TOPIC_ALIASES = {
    "tb": {"tb", "tuberculosis"},
    "tuberculosis": {"tb", "tuberculosis"},
    "pmjay": {"pmjay", "pm-jay", "ayushman", "bharat", "jan", "arogya"},
    "pm-jay": {"pmjay", "pm-jay", "ayushman", "bharat", "jan", "arogya"},
    "ayushman": {"pmjay", "pm-jay", "ayushman", "bharat", "jan", "arogya"},
    "jssk": {"jssk", "janani", "shishu", "suraksha", "karyakram"},
    "jsy": {"jsy", "janani", "suraksha", "yojana"},
    "himcare": {"himcare"},
    "nikshay": {"nikshay", "tb", "tuberculosis"},
    "dengue": {"dengue"},
    "chikungunya": {"chikungunya"},
    "zika": {"zika"},
    "stroke": {"stroke", "brain", "attack", "cerebrovascular"},
    "malaria": {"malaria"},
    "legionellosis": {"legionellosis", "legionnaires"},
    "cancer": {"cancer"},
}

KNOWN_TOPIC_TERMS = set(TOPIC_ALIASES)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9-]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _expand_aliases(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(TOPIC_ALIASES.get(token, set()))
    return expanded


def _extract_faq_title(result: RetrievalResult) -> str:
    if result.chunk_type != "faq" or not result.text:
        return ""

    first_line = result.text.splitlines()[0]
    return first_line.replace("FAQ:", "", 1).strip()


def _metadata_text(result: RetrievalResult) -> str:
    meta = result.metadata or {}
    fields = [
        meta.get("title", ""),
        meta.get("document_title", ""),
        meta.get("doc_title", ""),
        meta.get("faq_title", ""),
        meta.get("topic", ""),
        result.section_title,
        _extract_faq_title(result),
    ]
    return " ".join(str(field) for field in fields if field)


def metadata_match_adjustment(query: str, result: RetrievalResult) -> float:
    """
    Boost metadata topic matches and penalize clear topic mismatches.

    This runs after RRF and before final selection. It helps keep close but
    wrong neighbors out of the final context, e.g. TB -> Legionellosis,
    Dengue -> Chikungunya/Zika, PMJAY -> HIMCARE, Stroke -> Childhood Cancer.
    """
    query_tokens = _expand_aliases(_tokenize(query))
    meta_tokens = _expand_aliases(_tokenize(_metadata_text(result)))

    if not query_tokens or not meta_tokens:
        return 0.0

    overlap = query_tokens & meta_tokens
    overlap_ratio = len(overlap) / max(len(query_tokens), 1)
    adjustment = min(overlap_ratio * METADATA_BOOST_WEIGHT, METADATA_BOOST_WEIGHT)

    query_topic_terms = query_tokens & KNOWN_TOPIC_TERMS
    if query_topic_terms:
        meta_topic_terms = meta_tokens & KNOWN_TOPIC_TERMS
        if query_topic_terms & meta_topic_terms:
            adjustment += METADATA_BOOST_WEIGHT
        else:
            adjustment -= METADATA_PENALTY_WEIGHT

    return max(-METADATA_PENALTY_WEIGHT, min(adjustment, METADATA_BOOST_WEIGHT * 2))


def _normalise_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        value = 1.0 if len(scores) == 1 else 0.5
        return {key: value for key in scores}

    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def _clamp01(value: float) -> float:
    return max(0.0, min(value, 1.0))


# -- Cross-Encoder Reranker ---------------------------------------------------

class CrossEncoderReranker:
    """Lazy wrapper that loads the reranker only once per retriever instance."""

    def __init__(self, model_name: str = RERANKER_MODEL, enabled: bool = ENABLE_RERANKER):
        self.model_name = model_name
        self.enabled = enabled
        self.model = None
        self.load_failed = False

    def _load(self) -> None:
        if not self.enabled or self.model is not None or self.load_failed:
            return

        try:
            from sentence_transformers import CrossEncoder

            print(f"  [Reranker] Loading {self.model_name}...")
            self.model = CrossEncoder(self.model_name)
            print("  [Reranker] Ready.")
        except Exception as exc:
            self.load_failed = True
            print(f"  [Reranker] Disabled: {exc}")

    def score(self, query: str, results: list[RetrievalResult]) -> dict[str, float]:
        if not self.enabled or not results:
            return {}

        self._load()
        if self.model is None:
            return {}

        pairs = [(query, result.text) for result in results]
        try:
            raw_scores = self.model.predict(pairs)
        except Exception as exc:
            self.load_failed = True
            self.model = None
            print(f"  [Reranker] Disabled after scoring error: {exc}")
            return {}

        return {
            result.chunk_id: float(score)
            for result, score in zip(results, raw_scores)
        }


# -- RRF Merge ----------------------------------------------------------------

def rrf_merge(
    dense_ids: list[str],
    bm25_ids: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Returns:
        List of (chunk_id, rrf_score) sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])


# -- Hybrid Retriever ---------------------------------------------------------

class HybridRetriever:
    """
    Runs Dense + BM25 retrieval, merges with RRF, then reranks candidates.
    """

    def __init__(
        self,
        chunks_file: str = "all_chunks.jsonl",
        reranker_model: str = RERANKER_MODEL,
        enable_reranker: bool = ENABLE_RERANKER,
    ):
        print("[HybridRetriever] Initialising...")
        self.dense = DenseRetriever()
        self.bm25 = BM25Index(chunks_file=chunks_file)
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            enabled=enable_reranker,
        )
        print("[HybridRetriever] Ready.\n")

    def _chunk_matches_category(self, chunk_id: str, category: Optional[str]) -> bool:
        if category is None:
            return True

        chunk = self.bm25.get_chunk(chunk_id)
        if not chunk:
            return False

        return chunk.get("metadata", {}).get("category") == category

    def _result_for_chunk(
        self,
        chunk_id: str,
        score: float,
        dense_lookup: dict[str, RetrievalResult],
        bm25_lookup: dict[str, RetrievalResult],
    ) -> Optional[RetrievalResult]:
        if chunk_id in dense_lookup:
            r = dense_lookup[chunk_id]
        elif chunk_id in bm25_lookup:
            r = bm25_lookup[chunk_id]
        else:
            return None

        return RetrievalResult(
            chunk_id=r.chunk_id,
            chunk_type=r.chunk_type,
            section_title=r.section_title,
            text=r.text,
            score=score,
            metadata=r.metadata,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        fetch_k: int = TOP_K,
        verbose: bool = False,
    ) -> list[RetrievalResult]:
        """
        Run hybrid search and return top_k reranked results.

        The returned score is a reranked RRF-compatible score, not the raw
        cross-encoder logit. This preserves the verifier contract.
        """
        category = _route_query(query)

        if verbose:
            print(f"\n[Router] '{query}'")
            label = category if category else "ALL (cross-category)"
            print(f"[Router] -> category = '{label}'")

        dense_results = self.dense.retrieve(
            query=query,
            top_k=fetch_k,
            category_filter=category,
        )
        dense_ids = [r.chunk_id for r in dense_results]
        dense_lookup = {r.chunk_id: r for r in dense_results}

        if verbose:
            label = category if category else "ALL"
            print(f"\n[Dense] top {len(dense_results)} results (filtered: {label}):")
            for r in dense_results:
                print(f"  sim={r.score:.3f}  [{r.chunk_type}]  "
                      f"{r.section_title[:45]}  |  {r.metadata.get('title', '')[:30]}")

        bm25_raw_all = self.bm25.search(query, top_k=fetch_k * 3)
        bm25_raw = [
            (cid, score)
            for cid, score in bm25_raw_all
            if self._chunk_matches_category(cid, category)
        ][:fetch_k]

        bm25_ids = [cid for cid, _ in bm25_raw]
        bm25_lookup = {}
        for cid, score in bm25_raw:
            chunk = self.bm25.get_chunk(cid)
            if chunk:
                bm25_lookup[cid] = RetrievalResult(
                    chunk_id=cid,
                    chunk_type=chunk.get("chunk_type", ""),
                    section_title=chunk.get("section_title", ""),
                    text=chunk.get("text", ""),
                    score=score,
                    metadata=chunk.get("metadata", {}),
                )

        if verbose:
            label = category if category else "ALL"
            print(f"\n[BM25] top {len(bm25_raw)} results (filtered: {label}):")
            for cid, score in bm25_raw:
                chunk = self.bm25.get_chunk(cid)
                meta = chunk.get("metadata", {}) if chunk else {}
                print(f"  bm25={score:.3f}  [{chunk.get('chunk_type', '')}]  "
                      f"{chunk.get('section_title', '')[:45]}  |  "
                      f"{meta.get('title', '')[:30]}")

        merged = rrf_merge(dense_ids, bm25_ids, k=RRF_K)

        if verbose:
            print("\n[RRF] merged ranking:")
            for cid, rrf_score in merged[:RERANK_CANDIDATE_K]:
                in_dense = "D" if cid in dense_lookup else " "
                in_bm25 = "B" if cid in bm25_lookup else " "
                print(f"  rrf={rrf_score:.4f}  [{in_dense}{in_bm25}]  {cid[:55]}")

        candidates = []
        rrf_scores = {}
        for chunk_id, rrf_score in merged[:max(top_k, RERANK_CANDIDATE_K)]:
            result = self._result_for_chunk(chunk_id, rrf_score, dense_lookup, bm25_lookup)
            if result:
                candidates.append(result)
                rrf_scores[chunk_id] = rrf_score

        if not candidates:
            return []

        metadata_adjustments = {
            result.chunk_id: metadata_match_adjustment(query, result)
            for result in candidates
        }
        meta_adjusted_rrf = {
            result.chunk_id: _clamp01((rrf_scores[result.chunk_id] / MAX_RRF_SCORE) + metadata_adjustments[result.chunk_id])
            for result in candidates
        }

        reranker_raw = self.reranker.score(query, candidates)
        reranker_norm = _normalise_scores(reranker_raw)

        final_norm_scores = {}
        for result in candidates:
            cid = result.chunk_id
            if reranker_norm:
                final_norm = (
                    RERANKER_WEIGHT * reranker_norm.get(cid, 0.0)
                    + RRF_WEIGHT * meta_adjusted_rrf[cid]
                )
            else:
                final_norm = meta_adjusted_rrf[cid]
            final_norm_scores[cid] = _clamp01(final_norm)

        ranked_candidates = sorted(
            candidates,
            key=lambda result: final_norm_scores[result.chunk_id],
            reverse=True,
        )

        if verbose:
            print("\n[Metadata + Reranker] final ranking:")
            for result in ranked_candidates[:top_k]:
                cid = result.chunk_id
                print(
                    f"  final={final_norm_scores[cid]:.3f}  "
                    f"meta={metadata_adjustments[cid]:+.3f}  "
                    f"rerank={reranker_norm.get(cid, 0.0):.3f}  "
                    f"{cid[:55]}"
                )

        final_results = []
        for result in ranked_candidates[:top_k]:
            final_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                chunk_type=result.chunk_type,
                section_title=result.section_title,
                text=result.text,
                score=final_norm_scores[result.chunk_id] * MAX_RRF_SCORE,
                metadata=result.metadata,
            ))

        return final_results
