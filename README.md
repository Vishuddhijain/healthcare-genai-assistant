# Healthcare Awareness GenAI Assistant

A Retrieval-Augmented Generation (RAG) based Healthcare Awareness Chatbot designed to provide reliable, awareness-focused health information using trusted public healthcare sources. The system combines hybrid retrieval, semantic search, and reranking techniques to generate accurate, context-aware, and source-grounded responses.

---

## Project Objective

The goal of this project is to build a GenAI-powered Healthcare Awareness Assistant that answers general health-related queries using a curated healthcare knowledge base collected from trusted public healthcare sources.

The assistant focuses on:

- Preventive healthcare and wellness guidance
- Common symptoms and health awareness information
- Public health education and disease awareness
- Government healthcare schemes and beneficiary information
- General non-diagnostic healthcare education

**Note:** This chatbot is intended for healthcare awareness and educational purposes only. It does **not** provide medical diagnosis, prescriptions, medication dosage, or treatment recommendations. Users should consult qualified healthcare professionals for personalized medical advice.

---

## Data Sources

- WHO Fact Sheets
- Ministry of Health & Family Welfare (MoHFW)
- National Health Portal (NHP)
- National AIDS Control Organisation (NACO)
- National Institute of Mental Health and Neurosciences (NIMHANS)

---

## Key Features

- Hybrid Retrieval (Dense Retrieval + BM25)
- BGE-M3 Semantic Embeddings
- ChromaDB Vector Database
- BM25 Sparse Index
- Reciprocal Rank Fusion (RRF)
- Metadata-Aware Retrieval
- Cross-Encoder Reranking (BAAI/bge-reranker-base)
- Retrieval Verification
- Query Guard for Unsafe Queries
- Context-Aware Answer Generation
- Source-Grounded Responses

---

## Current Status

### Completed

- ✅ WHO Dataset Collection (231 Fact Sheets)
- ✅ Llama-based Data Structuring
- ✅ Text Cleaning & Normalization
- ✅ Metadata Generation
- ✅ Recursive Chunking
- ✅ BGE-M3 Embedding Generation
- ✅ ChromaDB Vector Store
- ✅ BM25 Sparse Index
- ✅ Hybrid Retrieval Pipeline
- ✅ Reciprocal Rank Fusion (RRF)
- ✅ Metadata-Aware Retrieval
- ✅ Cross-Encoder Reranking
- ✅ Retrieval Verification
- ✅ RAG Pipeline Integration

### In Progress

- 🚧 Healthcare Awareness Chatbot Interface
- 🚧 End-to-End Evaluation & Optimization
- 🚧 Deployment
