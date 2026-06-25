"""
answer_fusion.py
================
Combines retrieved chunks into one clean, deduplicated answer.

Architecture (group-aware fusion)
----------------------------------
1. Group chunks by document title (fallback to doc_id/section title).
2. Score each group against the query.
3. Keep groups whose score is > 60% of the best group's score.
4. Re-rank kept chunks by query relevance.
5. FAQ-FIRST only when a FAQ title directly matches the query AND the
   document title also matches the query topic.
6. Otherwise SECTION-FIRST.
7. Deduplicate sentences and build citations only from chunks that
   actually contributed content.

Constraints
-----------
- No LLM used here — pure text processing, no hallucination risk.
- Never adds medical facts not present in the chunks.
- Never truncates chunk text before sentence splitting.
- Adds a standard health disclaimer at the end of every answer.
- Max answer length: 1200 chars of content.
"""

import re
from collections import defaultdict


# ── Settings ──────────────────────────────────────────────────────────────────

MAX_ANSWER_CHARS = 1200
DEDUP_OVERLAP_RATIO = 0.60
GROUP_KEEP_RATIO = 0.60
FAQ_DIRECT_MATCH_THRESH = 0.50

DISCLAIMER = (
    "\n\n⚠ This information is for health awareness only. "
    "It is not medical advice, diagnosis, or treatment. "
    "Please consult a qualified healthcare professional."
)

STOPWORDS = {
    "what", "are", "the", "how", "does", "can", "i", "is", "in", "for",
    "a", "an", "to", "of", "do", "from", "get", "who", "apply", "my",
    "you", "will", "have", "be", "with", "on", "at", "by", "it", "its",
    "this", "that", "and", "or", "not", "during", "their", "available",
    "after", "still", "just", "person", "very", "every", "day", "days",
    "lot", "people", "also", "been", "when", "then", "where", "which",
    "there", "more", "some", "if", "up", "should", "would", "could",
    "need", "want", "about", "tell", "give", "me", "us", "them", "he",
    "she", "they", "we", "has", "had", "was", "were", "did", "being",
    "having"
}


# ── Text helpers ──────────────────────────────────────────────────────────────

def strip_chunk_prefix(text: str) -> str:
    """Remove Section:/FAQ: structural prefixes to get raw content."""
    text = re.sub(r"^Section:[^\n]*\n\n", "", text)
    text = re.sub(r"^FAQ:.*?\n", "", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """Split text into sentences for deduplication."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in raw:
        sub = [s.strip() for s in part.split("\n") if s.strip()]
        sentences.extend(sub)
    return [s for s in sentences if len(s) > 8]


def word_set(text: str) -> set:
    """Return lowercase words from text."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def content_word_set(query: str) -> set:
    """Query words minus stopwords."""
    return word_set(query) - STOPWORDS


def is_duplicate(sentence: str, seen_sentences: list) -> bool:
    """True if sentence overlaps too much with any already-seen sentence."""
    words = word_set(sentence)
    if not words:
        return True

    for seen in seen_sentences:
        seen_words = word_set(seen)
        if not seen_words:
            continue
        overlap = len(words & seen_words) / min(len(words), len(seen_words))
        if overlap >= DEDUP_OVERLAP_RATIO:
            return True
    return False


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _group_key(chunk: dict) -> str:
    """Prefer metadata title, then doc_id, then section title."""
    meta = chunk.get("metadata", {})
    return (
        meta.get("title")
        or meta.get("doc_id")
        or chunk.get("title")
        or chunk.get("doc_id")
        or chunk.get("section_title")
        or "unknown"
    )


def group_relevance_score(group_chunks: list, query: str) -> float:
    """
    Score a document group against the query using:
      60% content overlap + 40% title bonus
    """
    all_words = word_set(query)
    if not all_words:
        return 0.0

    cw = content_word_set(query)

    combined_text = " ".join(
        (c.get("section_title", "") + " " + c.get("text", ""))
        for c in group_chunks
    ).lower()
    combined_words = set(re.findall(r"\b\w+\b", combined_text))

    doc_title = group_chunks[0].get("metadata", {}).get("title", "").lower()
    doc_title_words = word_set(doc_title)

    # Fallback if query is mostly stopwords
    if not cw:
        return len(all_words & combined_words) / len(all_words)

    content_overlap = len(cw & combined_words) / len(cw)
    title_bonus = len(cw & doc_title_words) / len(cw)
    return 0.60 * content_overlap + 0.40 * title_bonus


def faq_directly_answers(faq_chunk: dict, query: str) -> bool:
    cw = content_word_set(query)
    if not cw:
        return False

    faq_title_words = word_set(faq_chunk.get("section_title", ""))
    faq_overlap = len(cw & faq_title_words) / len(cw)
    if faq_overlap < FAQ_DIRECT_MATCH_THRESH:
        return False

    doc_title = faq_chunk.get("metadata", {}).get("title", "").lower()
    doc_title_words = word_set(doc_title)
    return len(cw & doc_title_words) >= 1


def chunk_relevance_score(chunk: dict, query: str) -> float:
    """
    Per-chunk score used to rank chunks inside selected groups.
    """
    all_words = word_set(query)
    if not all_words:
        return 0.0

    cw = content_word_set(query)

    chunk_text = (chunk.get("section_title", "") + " " + chunk.get("text", "")).lower()
    chunk_words = set(re.findall(r"\b\w+\b", chunk_text))

    doc_title = chunk.get("metadata", {}).get("title", "").lower()
    doc_title_words = word_set(doc_title)

    full_overlap = len(all_words & chunk_words) / len(all_words)

    if not cw:
        return full_overlap

    content_overlap = len(cw & chunk_words) / len(cw)
    title_bonus = len(cw & doc_title_words) / len(cw)
    return 0.15 * full_overlap + 0.50 * content_overlap + 0.35 * title_bonus


# ── Citation builder ──────────────────────────────────────────────────────────

def build_citations(chunks: list) -> str:
    """Build a deduplicated citation list from chunk metadata."""
    seen_urls = {}
    citations = []
    idx = 1

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        url = meta.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls[url] = idx
            title = meta.get("title", "")
            source = meta.get("source_name", "")
            citations.append(f"[{idx}] {title} — {source}\n    {url}")
            idx += 1

    return "\n".join(citations) if citations else ""


# ── Core fusion ───────────────────────────────────────────────────────────────

def fuse_chunks(
    chunks: list,
    query: str = "",
    borderline_prefix: str = "",
) -> str:
    """
    Merge retrieved chunks into one clean answer.
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    # Step 1: group by source document
    grouped: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        grouped[_group_key(chunk)].append(chunk)

    # Step 2: score groups and keep only relevant groups
    if query:
        ranked_groups = sorted(
            ((key, group_relevance_score(group, query), group) for key, group in grouped.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        best_score = ranked_groups[0][1] if ranked_groups else 0.0
        threshold = best_score * GROUP_KEEP_RATIO

        kept_chunks = []
        for _, score, group in ranked_groups:
            if score > threshold:  # strict > is important for T02/T06 filtering
                kept_chunks.extend(group)

        if not kept_chunks and ranked_groups:
            kept_chunks.extend(ranked_groups[0][2])
    else:
        kept_chunks = list(chunks)

    if query:
        kept_chunks = sorted(
            kept_chunks,
            key=lambda c: chunk_relevance_score(c, query),
            reverse=True,
        )

    # Step 3: decide FAQ-first or SECTION-first
    direct_faq_chunks = [
        c for c in kept_chunks
        if c.get("chunk_type") == "faq" and faq_directly_answers(c, query)
    ] if query else []

    section_chunks = [c for c in kept_chunks if c.get("chunk_type") != "faq"]
    all_faq_chunks = [c for c in kept_chunks if c.get("chunk_type") == "faq"]

    if direct_faq_chunks:
        non_direct_faqs = [c for c in all_faq_chunks if c not in direct_faq_chunks]
        ordered_chunks = direct_faq_chunks + non_direct_faqs + section_chunks
    else:
        ordered_chunks = section_chunks + all_faq_chunks

    # Step 4: collect sentences, deduplicate
    seen_sentences = []
    content_parts = []
    total_chars = 0
    used_chunk_ids = set()
    current_section = None

    for chunk in ordered_chunks:
        raw = strip_chunk_prefix(chunk.get("text", ""))
        sents = split_sentences(raw)

        chunk_section = re.sub(
            r"\s*\(Part \d+/\d+\)\s*",
            "",
            chunk.get("section_title", "Information")
        ).strip()

        use_header = chunk.get("chunk_type") != "faq" and chunk_section != current_section
        added_in_this_chunk = []

        for s in sents:
            if total_chars >= MAX_ANSWER_CHARS:
                break
            if not is_duplicate(s, seen_sentences):
                seen_sentences.append(s)
                added_in_this_chunk.append(s)
                total_chars += len(s)

        if added_in_this_chunk:
            cid = chunk.get("chunk_id") or chunk.get("id") or str(id(chunk))
            used_chunk_ids.add(cid)

            if use_header:
                content_parts.append(f"\n{chunk_section}:")
                current_section = chunk_section

            if chunk.get("chunk_type") == "faq":
                content_parts.extend(added_in_this_chunk)
            else:
                content_parts.extend(f"  {s}" for s in added_in_this_chunk)

    if not seen_sentences:
        return "No relevant information could be extracted from the retrieved chunks."

    answer_body = "\n".join(content_parts).strip()

    # Step 5: cite only chunks that actually contributed content
    cited_chunks = []
    for chunk in ordered_chunks:
        cid = chunk.get("chunk_id") or chunk.get("id") or str(id(chunk))
        if cid in used_chunk_ids:
            cited_chunks.append(chunk)

    citations = build_citations(cited_chunks)
    citation_block = f"\n\nSources:\n{citations}" if citations else ""

    return (
        (borderline_prefix or "")
        + answer_body
        + citation_block
        + DISCLAIMER
    )