"""
retrieval.py
============
Dense vector retrieval using BGE-M3 + ChromaDB.

chunk_id fix
------------
Previously chunk_id was built as doc_id + '_' + section_title which:
  1. Created non-unique IDs (two 'Overview' sections in same doc = same ID)
  2. Never matched BM25 chunk_ids (which use real JSONL chunk_ids)
  3. Broke RRF merge in hybrid_retrieval.py

Fix: add 'ids' to ChromaDB include list and use raw['ids'][0] directly.
ChromaDB stores the original chunk_id as the document ID during embedding.

category_filter (NEW)
---------------------
retrieve() now accepts an optional category_filter string.
When provided, ChromaDB's where= clause restricts results to that category.

  category_filter="Government Healthcare Scheme"  → only scheme chunks
  category_filter="Disease & Awareness"           → only WHO/disease chunks
  category_filter=None                            → all chunks (old behaviour)

This is called by HybridRetriever after _route_query() classifies the query.
"""

from dataclasses import dataclass
from typing      import Optional
from sentence_transformers import SentenceTransformer
import chromadb


# ── Settings ──────────────────────────────────────────────────────────────────

EMBED_MODEL     = "BAAI/bge-m3"
CHROMA_DIR      = "../chroma_db"
COLLECTION_NAME = "health_and_govt_chunks"
TOP_K           = 10


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:      str     # real chunk_id e.g. 'who_malaria_chunk_3'
    chunk_type:    str
    section_title: str
    text:          str
    metadata:      dict
    score:         float   # cosine sim (dense), BM25 score, or RRF score


# ── Dense Retriever ───────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Embeds query with BGE-M3 and retrieves nearest chunks from ChromaDB.

    BGE-M3 asymmetric prefix:
      stored chunks → 'passage: ' prefix (added during embedding.py)
      user queries  → 'query: '   prefix (added here at retrieval time)
    """

    def __init__(self):
        print("  [DenseRetriever] Loading BGE-M3...")
        self.model      = SentenceTransformer(EMBED_MODEL)
        client          = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"  [DenseRetriever] Ready. {self.collection.count():,} vectors.")

    def _embed(self, query: str) -> list:
        return self.model.encode(
            "query: " + query,
            normalize_embeddings=True,
        ).tolist()

    def retrieve(
        self,
        query:           str,
        top_k:           int            = TOP_K,
        category_filter: Optional[str]  = None,   # ← NEW
    ) -> list:
        """
        Retrieve top_k chunks by cosine similarity.

        Args:
            query           : user question string
            top_k           : number of results to return
            category_filter : if set, restrict results to this ChromaDB
                              metadata category. One of:
                                "Government Healthcare Scheme"
                                "Disease & Awareness"
                              Pass None to search all chunks.

        Returns:
            list[RetrievalResult] sorted by similarity (highest first).
            chunk_id is the real ChromaDB document ID — matches BM25 chunk_ids
            so RRF merge in hybrid_retrieval.py works correctly.
        """
        query_embedding = self._embed(query)

        # Build the where clause only when a filter is requested
        where_clause = (
            {"category": category_filter}
            if category_filter is not None
            else None
        )

        raw = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            where            = where_clause,        # ← NEW: None = no filter
            include          = ["documents", "metadatas", "distances"],
        )

        results = []
        for chunk_id, doc, meta, dist in zip(
            raw["ids"][0],            # ← real chunk_id from ChromaDB
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(RetrievalResult(
                chunk_id      = chunk_id,          # e.g. 'who_malaria_chunk_3'
                chunk_type    = meta.get("chunk_type", ""),
                section_title = meta.get("section_title", ""),
                text          = doc,
                score         = 1.0 - dist,        # distance → similarity
                metadata      = {
                    "doc_id":      meta.get("doc_id", ""),
                    "title":       meta.get("title", ""),
                    "category":    meta.get("category", ""),
                    "topic":       meta.get("topic", ""),
                    "source_name": meta.get("source_name", ""),
                    "source_url":  meta.get("source_url", ""),
                },
            ))

        return results