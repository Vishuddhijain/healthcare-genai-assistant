## System Architecture

### 1. Data Ingestion & Indexing Pipeline

```text
WHO Fact Sheets (TXT Files)
           │
           ▼
Llama Model (Information Extraction & Structuring)
           │
           ▼
      Structured JSON
           │
           ▼
Text Cleaning & Normalization
           │
           ▼
   Metadata Generation
           │
           ▼
   Recursive Chunking
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
Generate BGE-M3 Embeddings      Build BM25 Index
     │                              │
     ▼                              ▼
ChromaDB (Vector Store)      Sparse Text Index
```

---

### 2. Query & Retrieval Pipeline

```text
                       User Query
                           │
                           ▼
                      Query Guard
                           │
                           ▼
                 Query Category Routing
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
 Dense Retrieval (BGE-M3)          BM25 Retrieval
 (Search ChromaDB)                (Search BM25 Index)
         │                                   │
         └─────────────────┬─────────────────┘
                           ▼
                 Hybrid Retrieval (RRF)
                           │
                           ▼
             Metadata-Aware Score Adjustment
                           │
                           ▼
 Cross-Encoder Reranker (BAAI/bge-reranker-base)
                           │
                           ▼
               Retrieval Verification
                           │
                           ▼
        Answer Generation & Source Attribution
                           │
                           ▼
          Healthcare Awareness Chatbot
```
