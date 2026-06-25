"""
bm25_retrieval.py
=================
BM25 keyword retrieval over the chunked knowledge base.

Why BM25 alongside dense retrieval?
-------------------------------------
BGE-M3 (dense) is great at semantic matching:
  "cardiac arrest" matches "heart stops beating"
But it can miss exact keyword matches:
  "PM-JAY scheme" or "JSSK" or a specific scheme name

BM25 is great at exact keyword matching but misses semantics.
Hybrid search (dense + BM25 + RRF) gets the best of both.

BM25 score formula:
  sum over query terms t of:
    IDF(t) * TF(t,d) * (k1+1) / (TF(t,d) + k1*(1-b + b*|d|/avgdl))

  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  k1 = 1.5  (term frequency saturation)
  b  = 0.75 (document length normalisation)

Index
-----
Built once at startup from all_chunks.jsonl, held in memory.
chunk_ids from JSONL match ChromaDB document IDs exactly —
this is what makes RRF merge work correctly.
"""

import json
import re
import math
import os
from collections import Counter


# ── BM25 hyperparameters ──────────────────────────────────────────────────────

K1 = 1.5
B  = 0.75

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","by","from","is","are","was","were","be","been",
    "has","have","had","do","does","did","will","would","can","could",
    "this","that","these","those","it","its","as","if","so","not",
    "also","may","should","into","than","more","about","which","when",
}


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index built from all_chunks.jsonl.

    chunk_ids here are the real chunk_ids from the JSONL file
    (e.g. 'who_malaria_chunk_3') — same IDs stored in ChromaDB.
    This ensures chunk_ids match between Dense and BM25 results
    so RRF merge correctly identifies chunks that appear in both lists.
    """

    def __init__(self, chunks_file: str = "all_chunks.jsonl"):
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        print("  [BM25Index] Building index...")
        self._build(chunks_file)
        print(f"  [BM25Index] Ready. {self.N:,} documents indexed.")

    def _build(self, chunks_file: str) -> None:
        self.chunk_ids     = []   # list of chunk_id strings
        self.chunks_lookup = {}   # chunk_id → full chunk dict
        self.corpus_tokens = []   # tokenized text per chunk (parallel to chunk_ids)

        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                cid   = chunk["chunk_id"]   # real chunk_id — matches ChromaDB ID

                self.chunk_ids.append(cid)
                self.chunks_lookup[cid] = chunk
                self.corpus_tokens.append(tokenize(chunk["text"]))

        self.N     = len(self.chunk_ids)
        doc_lens   = [len(t) for t in self.corpus_tokens]
        self.avgdl = sum(doc_lens) / self.N if self.N else 1.0

        # Document frequency: number of docs containing each term
        self.df: dict = Counter()
        for tokens in self.corpus_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Search the index for the top_k most relevant chunks.

        Returns:
            List of (chunk_id, bm25_score) sorted by score descending.
            chunk_id matches ChromaDB document IDs exactly.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: dict = {}

        for term in q_tokens:
            if term not in self.df:
                continue
            idf_val = self._idf(term)
            for doc_idx, tokens in enumerate(self.corpus_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl      = len(tokens)
                tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / self.avgdl))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf_val * tf_norm

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.chunk_ids[i], score) for i, score in ranked]

    def get_chunk(self, chunk_id: str) -> dict:
        """Return full chunk dict for a given chunk_id."""
        return self.chunks_lookup.get(chunk_id, {})    