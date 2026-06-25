# Dataset & Development Plan

## Phase 1 – Dataset Collection

**Task:** Collect WHO Health Fact Sheets

**Status:** ✅ Completed

**Output:**

- 231 WHO Fact Sheet TXT documents

---

## Phase 2 – Data Structuring

**Task:** Convert raw TXT documents into structured JSON using the Llama model

**Status:** ✅ Completed

**Output:**

- Structured JSON documents
- Standardized sections (Overview, Symptoms, Prevention, FAQ, etc.)

---

## Phase 3 – Metadata Generation

**Task:** Generate metadata for each document

**Status:** ✅ Completed

**Output:**

- Document title
- Category
- Section title
- Chunk type
- Source information

---

## Phase 4 – Data Preprocessing & Chunking

**Task:** Clean, normalize, and recursively chunk the documents

**Status:** ✅ Completed

**Output:**

- High-quality text chunks
- Metadata-linked chunks
- Optimized chunk sizes for retrieval

---

## Phase 5 – Vector Embedding Generation

**Task:** Generate dense embeddings using BGE-M3

**Status:** ✅ Completed

**Output:**

- Dense vector embeddings for all chunks

---

## Phase 6 – Hybrid Retrieval Index

**Task:** Build retrieval indexes

**Status:** ✅ Completed

**Output:**

- ChromaDB Vector Store
- BM25 Sparse Index

---

## Phase 7 – Retrieval Pipeline

**Task:** Implement an advanced hybrid retrieval pipeline

**Status:** ✅ Completed

**Features:**

- Query Guard
- Query Category Routing
- Dense Retrieval (BGE-M3)
- BM25 Retrieval
- Reciprocal Rank Fusion (RRF)
- Metadata-Aware Filtering
- Cross-Encoder Reranking (BAAI/bge-reranker-base)
- Retrieval Verification
- Answer Fusion

---

## Phase 8 – Healthcare Awareness Chatbot

**Task:** Build the end-to-end RAG-powered Healthcare Awareness Chatbot

**Status:** 🚧 In Progress

**Features:**

- Retrieval-Augmented Generation (RAG)
- Context-aware responses
- Source-grounded answers
- Safety guardrails
- Healthcare awareness assistance
