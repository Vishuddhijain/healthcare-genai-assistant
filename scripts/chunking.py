"""
chunking.py
===========
Recursive chunking pipeline for WHO health and Government scheme JSONs.
Outputs a single all_chunks.jsonl combining both sources.

How the data is divided
-----------------------
Each JSON document has a "content" list of sections, e.g.:

  WHO Abortion document:
    [Key facts] [Overview] [Scope of the problem] [FAQ: Is it legal?] ...

  Govt Scheme document:
    [Overview] [Eligibility Criteria] [Benefits] [FAQ: Who can apply?] ...

For each section we do one of two things:

  1. FAQ section  (title starts with "FAQ:")
     → Always kept as ONE chunk. The question + answer stay together.
       Never split, because splitting a Q from its A destroys retrieval quality.

  2. Regular section
     → If text <= CHUNK_SIZE: one chunk, done.
     → If text >  CHUNK_SIZE: recursively split into parts.
       Each part carries the section title so it is self-contained.
       CHUNK_OVERLAP chars of the previous part's tail are prepended to the
       next part so no sentence is lost at the boundary.
       Overlap always snaps to a word boundary (never cuts mid-word).

Chunk size choice
-----------------
CHUNK_SIZE = 1000 chars ≈ 250-350 tokens for English text.
BGE-M3 limit is 512 tokens. 1000 chars leaves safe headroom after
adding the "Section: <title>" prefix (~30-50 chars).

Normalization (runs before splitting)
--------------------------------------
Strips noise that wastes token slots in BGE-M3:
  - Broken Rupee symbol [?] → ₹
  - Markdown: **bold**, ### headers, > blockquotes, `code`
  - Citation numbers [1] [22] (1) (22)
  - Windows \r line endings, \t tabs
  - Multiple spaces and excessive blank lines

Metadata on every chunk (exactly these six fields):
  doc_id, title, category, topic, source_name, source_url
"""

import json
import os
import re

# ── Settings ──────────────────────────────────────────────────────────────────

WHO_FILE  = "../who_structured_master.json"
GOVT_FILE = "../govt_structured_master.json"
OUTPUT    = "../all_chunks.jsonl"

CHUNK_SIZE    = 1200   # chars — keeps chunks ~250-350 tokens, safe under BGE-M3's 512 limit
CHUNK_OVERLAP = 100    # chars — overlap between consecutive parts, snapped to word boundary

SKIP_VALUES = {
    "not applicable",
    "information disclosure",
    "n/a",
}

# Malformed section titles where the title text ran into the content
# (data issue in source JSON — strip these down to a clean title)
TITLE_MAX_LEN = 80   # any section_title longer than this is treated as malformed


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Remove noise that wastes BGE-M3 token slots without adding meaning.
    Only called on section text — never on titles or metadata.
    """
    if not text:
        return ""

    # Fix broken Rupee encoding artifact
    text = text.replace("[?]", "₹")

    # Remove citation numbers  [1] [22]  and  (1) (22)
    text = re.sub(r"\s*\[\d+\]\s*", " ", text)
    text = re.sub(r"\s*\(\d+\)\s*", " ", text)

    # Strip markdown syntax — BGE-M3 is not markdown-aware
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)          # **bold** *italic*
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)   # ### Heading
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)        # > blockquote
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text, flags=re.DOTALL)  # `code`

    # Normalise line endings and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # single \n → space, keep \n\n
    text = re.sub(r" {2,}", " ", text)              # collapse multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse excessive blank lines

    return text.strip()


def clean_section_title(title: str) -> str:
    """
    Guard against malformed section titles where content leaked into the title field.
    If title is longer than TITLE_MAX_LEN, truncate at the first sentence boundary.
    """
    title = title.strip()
    if len(title) <= TITLE_MAX_LEN:
        return title
    # Find first sentence end within a reasonable length
    cut = title.find(". ", 20)
    if cut != -1 and cut < TITLE_MAX_LEN:
        return title[:cut].strip()
    return title[:TITLE_MAX_LEN].strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_faq(section_title: str) -> bool:
    return section_title.strip().upper().startswith("FAQ:")


def is_placeholder(text: str) -> bool:
    t = text.strip().lower()
    return not t or any(t.startswith(v) for v in SKIP_VALUES)


def build_metadata(doc: dict) -> dict:
    return {
        "doc_id":      doc.get("doc_id"),
        "title":       doc.get("title"),
        "category":    doc.get("category", "General"),
        "topic":       doc.get("topic", "General"),
        "source_name": doc.get("source_name"),
        "source_url":  doc.get("source_url"),
    }


def snap_to_word_boundary(text: str, pos: int) -> int:
    """
    Move pos forward to the next space so the overlap never starts mid-word.
    E.g. 'Comprehen|sive' → moves to 'Comprehensive |' 
    """
    while pos < len(text) and text[pos] != " ":
        pos += 1
    return pos


# ── Recursive Text Splitter ───────────────────────────────────────────────────

def recursive_split(text: str) -> list:
    """
    Split text into chunks of at most CHUNK_SIZE characters.

    Priority order for split points (highest = most preferred):
      1. Paragraph break (\n\n)  — keeps full paragraphs together
      2. Newline (\n)            — keeps lines together
      3. Sentence end (". ")     — keeps sentences together
      4. Word boundary (" ")     — last resort before character split

    Overlap:
      The last CHUNK_OVERLAP chars of each part are prepended to the next
      part so context is preserved at boundaries. The overlap start is always
      snapped to a word boundary so no word is ever cut in half.

    Returns a list of text strings, each <= CHUNK_SIZE chars.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    result = []

    def _split(segment: str, sep_idx: int) -> None:
        if len(segment) <= CHUNK_SIZE or sep_idx >= len(separators):
            if segment.strip():
                result.append(segment.strip())
            return

        sep   = separators[sep_idx]
        parts = segment.split(sep)
        buf   = ""

        for part in parts:
            candidate = buf + (sep if buf else "") + part
            if len(candidate) <= CHUNK_SIZE:
                buf = candidate
            else:
                if buf.strip():
                    _split(buf.strip(), sep_idx + 1)

                # Build overlap: take last CHUNK_OVERLAP chars, then snap to word boundary
                if CHUNK_OVERLAP and buf:
                    raw_start = max(0, len(buf) - CHUNK_OVERLAP)
                    word_start = snap_to_word_boundary(buf, raw_start)
                    overlap_tail = buf[word_start:].strip()
                else:
                    overlap_tail = ""

                # Start next buffer with overlap context prepended
                buf = (overlap_tail + " " + part).strip() if overlap_tail else part

        if buf.strip():
            _split(buf.strip(), sep_idx + 1)

    _split(text, 0)
    return result if result else [text]


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(data: list) -> list:
    """
    Convert a list of structured documents into retrieval-ready chunks.

    FAQ sections  → one chunk per Q&A pair (never split).
    Other sections → one chunk if short, recursively split if long.
    Every chunk is self-contained: section title is embedded in the text.
    """
    chunks = []

    for doc in data:
        meta    = build_metadata(doc)
        counter = 1

        for section in doc.get("content", []):
            raw_title = section.get("section_title", "").strip()
            raw_text  = section.get("text", "") or ""

            title = clean_section_title(raw_title)
            text  = normalize(raw_text)

            if is_placeholder(text):
                continue

            # ── FAQ: never split ─────────────────────────────────────────────
            if is_faq(title):
                question = title[4:].strip()    # drop "FAQ:" prefix from title
                chunks.append({
                    "chunk_id":      f"{meta['doc_id']}_chunk_{counter}",
                    "chunk_type":    "faq",
                    "section_title": title,
                    "text":          f"FAQ: {question}\n{text}",
                    "metadata":      meta,
                })
                counter += 1

            # ── Regular section: split if long ───────────────────────────────
            else:
                parts = recursive_split(text)

                for i, part in enumerate(parts, start=1):
                    if len(parts) == 1:
                        chunk_text = f"Section: {title}\n\n{part}"
                        chunk_type = "section"
                    else:
                        chunk_text = f"Section: {title} (Part {i}/{len(parts)})\n\n{part}"
                        chunk_type = "section_part"

                    chunks.append({
                        "chunk_id":      f"{meta['doc_id']}_chunk_{counter}",
                        "chunk_type":    chunk_type,
                        "section_title": title,
                        "text":          chunk_text,
                        "metadata":      meta,
                    })
                    counter += 1

    return chunks


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  loaded  {len(data):>4} docs  ← {path}")
    return data


def save_jsonl(chunks: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  saved   {len(chunks):,} chunks → {path}")


def print_stats(chunks: list, label: str) -> None:
    faq   = sum(1 for c in chunks if c["chunk_type"] == "faq")
    sec   = sum(1 for c in chunks if c["chunk_type"] == "section")
    parts = sum(1 for c in chunks if c["chunk_type"] == "section_part")
    lens  = [len(c["text"]) for c in chunks]
    avg   = sum(lens) // len(lens) if lens else 0
    print(f"\n  {label}")
    print(f"    total        : {len(chunks):,}")
    print(f"    faq          : {faq:,}")
    print(f"    section      : {sec:,}")
    print(f"    section_part : {parts:,}")
    print(f"    avg len      : {avg:,} chars  (~{avg//4} tokens)")
    print(f"    max len      : {max(lens):,} chars  (~{max(lens)//4} tokens)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  CHUNKING PIPELINE")
    print(f"  chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}")
    print("=" * 55)

    print("\n[1] Loading...")
    who_data  = load_json(WHO_FILE)
    govt_data = load_json(GOVT_FILE)

    print("\n[2] Chunking...")
    who_chunks  = chunk_documents(who_data)
    govt_chunks = chunk_documents(govt_data)
    all_chunks  = who_chunks + govt_chunks

    print_stats(who_chunks,  "WHO")
    print_stats(govt_chunks, "GOVT")

    lens = [len(c["text"]) for c in all_chunks]
    print(f"\n  COMBINED")
    print(f"    total chunks : {len(all_chunks):,}")
    print(f"    avg len      : {sum(lens)//len(lens):,} chars")
    print(f"    max len      : {max(lens):,} chars  (~{max(lens)//4} tokens)")

    print("\n[3] Saving...")
    save_jsonl(all_chunks, OUTPUT)

    # Sanity checks
    over  = sum(1 for c in all_chunks if len(c["text"]) // 4 > 512)
    dups  = len(all_chunks) - len({c["chunk_id"] for c in all_chunks})
    empty = sum(1 for c in all_chunks if not c["text"].strip())
    

    print(f"\n  Sanity checks:")
    print(f"    chunks > 512 tokens (chars/4) : {over}")
    print(f"    duplicate chunk_ids           : {dups}")
    print(f"    empty chunks                  : {empty}")
    

    # Sample — show a split section to verify overlap is clean
    split_samples = [c for c in all_chunks if c["chunk_type"] == "section_part"]
    if split_samples:
        print(f"\n  Sample split chunk (Part 1):")
        c = split_samples[0]
        lines = c["text"].split("\n\n", 1)
        print(f"    chunk_id : {c['chunk_id']}")
        print(f"    header   : {lines[0]}")
        print(f"    text[0:120]: {lines[1][:120] if len(lines)>1 else ''}")
        # Find the next part
        next_id = c["chunk_id"].replace(
            f"_chunk_{c['chunk_id'].split('_chunk_')[1]}",
            f"_chunk_{int(c['chunk_id'].split('_chunk_')[1])+1}"
        )
        nxt = next((x for x in all_chunks if x["chunk_id"] == next_id), None)
        if nxt:
            nxt_lines = nxt["text"].split("\n\n", 1)
            print(f"\n  Same section, next part:")
            print(f"    chunk_id : {nxt['chunk_id']}")
            print(f"    header   : {nxt_lines[0]}")
            print(f"    text[0:120]: {nxt_lines[1][:120] if len(nxt_lines)>1 else ''}")

    print(f"\n{'='*55}")
    print(f"  Done → {OUTPUT}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
