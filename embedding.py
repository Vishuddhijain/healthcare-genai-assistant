"""
embedding.py
============
Embeds all chunks from all_chunks.jsonl using BGE-M3
and stores them in a persistent ChromaDB collection.

Install before running:
    pip install sentence-transformers chromadb

BGE-M3 notes:
  - Model: BAAI/bge-m3  (downloaded automatically on first run ~2.2 GB)
  - Max input: 8192 tokens  (our chunks are ~250-350 tokens — well within limit)
  - Embedding dim: 1024
  - Best practice: prefix non-FAQ text with "passage: " for retrieval tasks
    (BGE models are trained with this prefix for asymmetric search)

ChromaDB notes:
  - Stored locally in ./chroma_db/ folder (created automatically)
  - Collection name: health_and_govt_chunks
  - Metadata stored alongside each vector for filtering and citation display
  - Batching: 64 chunks per batch to avoid memory issues with BGE-M3
"""

import json
import os
import time

from sentence_transformers import SentenceTransformer  # type: ignore[import]
import chromadb  # type: ignore[import]


# ── Settings ──────────────────────────────────────────────────────────────────

CHUNKS_FILE     = "all_chunks.jsonl"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "health_and_govt_chunks"
EMBED_MODEL     = "BAAI/bge-m3"
BATCH_SIZE      = 64    

# ── Load Chunks ───────────────────────────────────────────────────────────────

def load_chunks(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found: {path}")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks):,} chunks from {path}")
    return chunks


# ── Prepare for ChromaDB ──────────────────────────────────────────────────────

def prepare_chroma_fields(chunks: list):
    """
    Extract the four parallel lists ChromaDB's add() expects:
      ids, documents, metadatas, (embeddings are passed separately)

    ChromaDB rules:
      - ids        : must be unique strings
      - documents  : the raw text stored alongside the vector
      - metadatas  : dict of str/int/float/bool values only — no None, no lists
      - embeddings : list of float vectors (added in batches below)
    """
    ids        = []
    documents  = []
    metadatas  = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])

        meta = chunk["metadata"]
        metadatas.append({
            "doc_id":        meta.get("doc_id")      or "",
            "title":         meta.get("title")        or "",
            "category":      meta.get("category")     or "General",
            "topic":         meta.get("topic")        or "General",
            "source_name":   meta.get("source_name")  or "",
            "source_url":    meta.get("source_url")   or "",
            # chunk-level fields useful for filtering and display
            "chunk_type":    chunk.get("chunk_type")   or "",
            "section_title": chunk.get("section_title") or "",
        })

    return ids, documents, metadatas


def prepare_texts_for_embedding(chunks: list) -> list:
    """
    BGE-M3 retrieval best practice:
      - Passages (stored chunks) are prefixed with "passage: "
      - Queries (user questions) are prefixed with "query: "
    This asymmetric prefix was used during BGE-M3's training and
    improves retrieval accuracy for question-answering tasks.

    FAQ chunks already start with "FAQ: <question>\n<answer>" —
    they are kept as-is because the question itself acts as the anchor.
    """
    texts = []
    for chunk in chunks:
        if chunk["chunk_type"] == "faq":
            texts.append(chunk["text"])           # FAQ: keep as-is
        else:
            texts.append("passage: " + chunk["text"])   # section/section_part
    return texts


# ── Embed + Store ─────────────────────────────────────────────────────────────

def embed_and_store(chunks: list, model: SentenceTransformer, collection) -> None:
    """
    Embed chunks in batches and upsert into ChromaDB.
    Uses upsert (not add) so re-running the script is safe —
    existing chunks are updated, new ones are added, nothing duplicated.
    """
    ids, documents, metadatas = prepare_chroma_fields(chunks)
    texts_for_embedding       = prepare_texts_for_embedding(chunks)

    total   = len(chunks)
    n_batch = (total + BATCH_SIZE - 1) // BATCH_SIZE   # ceil division

    print(f"\n  Embedding and storing {total:,} chunks in {n_batch} batches...")
    t_start = time.time()

    for batch_idx in range(n_batch):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)

        batch_texts     = texts_for_embedding[start:end]
        batch_ids       = ids[start:end]
        batch_docs      = documents[start:end]
        batch_metas     = metadatas[start:end]

        # Embed — normalize_embeddings=True is required for cosine similarity search
        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Upsert into ChromaDB
        collection.upsert(
            ids        = batch_ids,
            embeddings = embeddings,
            documents  = batch_docs,
            metadatas  = batch_metas,
        )

        elapsed = time.time() - t_start
        done    = end
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_idx+1:>3}/{n_batch}  "
            f"chunks {start+1}-{end}  "
            f"[{done*100//total:>3}%]  "
            f"elapsed {elapsed:>6.1f}s  "
            f"ETA {eta:>6.1f}s"
        )

    print(f"\n  Done. Total time: {time.time()-t_start:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  EMBEDDING PIPELINE  →  ChromaDB")
    print(f"  model      : {EMBED_MODEL}")
    print(f"  batch_size : {BATCH_SIZE}")
    print(f"  chroma_dir : {CHROMA_DIR}")
    print(f"  collection : {COLLECTION_NAME}")
    print("=" * 55)

    # 1. Load chunks
    print("\n[1] Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)

    faq   = sum(1 for c in chunks if c["chunk_type"] == "faq")
    sec   = sum(1 for c in chunks if c["chunk_type"] == "section")
    parts = sum(1 for c in chunks if c["chunk_type"] == "section_part")
    print(f"    faq={faq:,}  section={sec:,}  section_part={parts:,}")

    # 2. Load BGE-M3
    print(f"\n[2] Loading {EMBED_MODEL}...")
    print("    (First run downloads ~2.2 GB — subsequent runs use local cache)")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"    Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # 3. Connect to ChromaDB
    print(f"\n[3] Connecting to ChromaDB at {CHROMA_DIR}...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity (matches normalize_embeddings=True)
    )
    existing = collection.count()
    print(f"    Collection '{COLLECTION_NAME}' — existing vectors: {existing:,}")

    # 4. Embed and store
    print("\n[4] Embedding and storing...")
    embed_and_store(chunks, model, collection)

    # 5. Verify
    print("\n[5] Verification...")
    final_count = collection.count()
    print(f"    Vectors in collection: {final_count:,}")
    print(f"    Expected             : {len(chunks):,}")
    if final_count == len(chunks):
        print("    ✓ All chunks stored successfully")
    else:
        print(f"    ⚠ Mismatch — {len(chunks) - final_count} chunks may have failed")

    

    print(f"\n{'='*55}")
    print(f"  Done. ChromaDB stored at: {CHROMA_DIR}/")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Vectors    : {final_count:,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()