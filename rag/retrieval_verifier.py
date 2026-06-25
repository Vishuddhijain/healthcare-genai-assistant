"""
retrieval_verifier.py
=====================
Verifies whether retrieved chunks are good enough to answer a query.
If not, returns a safe fallback instead of hallucinating.

How it works
------------
ChromaDB returns cosine DISTANCE (0.0 = identical, 2.0 = opposite).
We convert to similarity = 1 - distance.

Thresholds (tuned for BGE-M3 on health/govt data):
  similarity >= 0.60  →  CONFIDENT   → proceed to answer fusion
  similarity >= 0.50  →  BORDERLINE  → answer with caution disclaimer
  similarity <  0.35  →  NOT FOUND   → return safe fallback, do not answer
"""

from dataclasses import dataclass
from typing import Optional


# ── Thresholds ────────────────────────────────────────────────────────────────

CONFIDENT_THRESHOLD  = 0.60   # top chunk similarity must be >= this
BORDERLINE_THRESHOLD = 0.50   # between this and CONFIDENT = answer with disclaimer
MIN_CHUNKS_REQUIRED  = 1      # at least this many chunks must meet BORDERLINE


# ── Fallback messages ─────────────────────────────────────────────────────────

FALLBACK_NOT_FOUND = (
    "I could not find reliable information on this topic in the current "
    "knowledge base (WHO Fact Sheets and Government Healthcare Schemes). "
    "Please consult a qualified healthcare professional for medical advice."
)

FALLBACK_BORDERLINE = (
    "The following information is based on partial matches in the knowledge base. "
    "Please verify with a qualified healthcare professional before acting on it.\n\n"
)

FALLBACK_OFF_TOPIC = (
    "This topic does not appear to be covered in the available sources "
    "(WHO health awareness content and Indian Government healthcare schemes). "
    "For medical concerns, please consult a doctor or call the national health helpline 104."
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    status: str                    # "confident" | "borderline" | "not_found"
    top_similarity: float          # similarity score of best chunk
    valid_chunks: list             # chunks that passed the threshold
    fallback_message: Optional[str]  # set when status != "confident"


# ── Main verifier ─────────────────────────────────────────────────────────────

def verify_retrieval(
    query: str,
    retrieved_chunks: list,
    distances: list,
) -> VerificationResult:
    """
    Check whether retrieved chunks are relevant enough to answer the query.

    Args:
        query           : the user's original question (for logging)
        retrieved_chunks: list of chunk dicts from ChromaDB
                          each has keys: chunk_id, text, metadata, chunk_type, section_title
        distances       : list of cosine distances from ChromaDB (same order as chunks)

    Returns:
        VerificationResult with status, valid_chunks, and fallback_message
    """
    if not retrieved_chunks or not distances:
        return VerificationResult(
            status           = "not_found",
            top_similarity   = 0.0,
            valid_chunks     = [],
            fallback_message = FALLBACK_NOT_FOUND,
        )

    # Convert distances to similarities
    similarities = [1.0 - d for d in distances]
    top_similarity = similarities[0]

    # Pair chunks with their similarity scores
    paired = list(zip(retrieved_chunks, similarities))

    # Hard not-found: best chunk is below borderline threshold
    if top_similarity < BORDERLINE_THRESHOLD:
        return VerificationResult(
            status           = "not_found",
            top_similarity   = top_similarity,
            valid_chunks     = [],
            fallback_message = FALLBACK_NOT_FOUND,
        )

    # Filter: keep only chunks above the borderline threshold
    valid = [(c, s) for c, s in paired if s >= BORDERLINE_THRESHOLD]

    if len(valid) < MIN_CHUNKS_REQUIRED:
        return VerificationResult(
            status           = "not_found",
            top_similarity   = top_similarity,
            valid_chunks     = [],
            fallback_message = FALLBACK_NOT_FOUND,
        )

    valid_chunks = [c for c, s in valid]

    # Confident: top chunk clears the confident threshold
    if top_similarity >= CONFIDENT_THRESHOLD:
        return VerificationResult(
            status           = "confident",
            top_similarity   = top_similarity,
            valid_chunks     = valid_chunks,
            fallback_message = None,
        )

    # Borderline: above minimum but below confident
    return VerificationResult(
        status           = "borderline",
        top_similarity   = top_similarity,
        valid_chunks     = valid_chunks,
        fallback_message = FALLBACK_BORDERLINE,
    )
