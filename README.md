# Healthcare Awareness GenAI Assistant

A Retrieval-Augmented Generation (RAG) based Healthcare Awareness Chatbot designed to provide reliable, awareness-focused health information using trusted public healthcare sources.

## Project Objective

The goal of this project is to build a GenAI-powered healthcare awareness assistant that can answer general health-related queries using a curated healthcare knowledge base collected from reliable public sources.

The assistant focuses on:

* Preventive care and wellness guidance
* Common symptoms and health awareness information
* Public health awareness topics
* Government healthcare schemes and resources
* General non-diagnostic healthcare education

**Note:** This chatbot does not provide medical diagnosis, prescriptions, medicine dosage, or treatment recommendations. Users are encouraged to consult qualified healthcare professionals for medical advice.

---

## Data Sources

The healthcare knowledge base is being built using reliable public healthcare sources, including:

* World Health Organization (WHO) Fact Sheets
* Ministry of Health & Family Welfare (MoHFW), India
* National Health Portal (NHP), India
* National AIDS Control Organisation (NACO)
* National Institute of Mental Health and Neurosciences (NIMHANS)
* Other legally usable public healthcare resources

All sources are documented in `SOURCES.md`.

---

## Data Processing Pipeline

Raw Healthcare Data
↓
Data Collection
↓
Data Cleaning & Standardization
↓
Llama-based Structured JSON Generation
↓
Metadata Generation
↓
Document Structure-Based Chunking
↓
Embeddings Generation
↓
Vector Database (FAISS)
↓
RAG Pipeline
↓
Healthcare Awareness Chatbot

---

## Structured Knowledge Base Schema

Each healthcare document is converted into a standardized JSON format containing:

* Document ID
* Title
* Source Information
* Topic
* Category
* Overview
* Types
* Symptoms
* Warning Signs
* Risk Factors
* Prevention
* Environmental Control
* Vaccination
* Diagnosis (Awareness Only)
* Treatment Information (Awareness Only)
* When to See a Doctor
* India-Specific Context
* Safety Notes

---

## Metadata Management

Metadata is maintained for every document and chunk, including:

* Source Name
* Source URL
* Topic
* Category
* Document ID
* Retrieval Date
* Chunk Information

This enables source citation, filtering, and traceability during retrieval.

---

## RAG Architecture

### Selected RAG Strategy

* Simple RAG with Memory

### Chunking Strategy

* Document Structure-Based Chunking
* Recursive Chunking for oversized sections

This approach was selected because the healthcare documents are already organized into meaningful sections such as symptoms, prevention, and warning signs, improving retrieval accuracy.

---

## Current Status

### Completed

* Data Source Identification
* Healthcare Data Collection
* Knowledge Base Organization
* Structured JSON Validation
* Metadata Design
* Private GitHub Repository Setup
* RAG Architecture Selection
* Chunking Strategy Selection

### In Progress
* Chunk Generation

### Planned

* Embeddings Generation
* FAISS Vector Database Integration
* Retrieval Pipeline Development
* LLM Integration
* Frontend Development
* Source Citation System
* Safety Guardrails
* Deployment

---

## Tech Stack

### Frontend

* React.js / Next.js
* Tailwind CSS

### Backend

* Python (FastAPI)

### AI & LLM

* Llama
* Groq API (if approved)

### Vector Database

* FAISS

### Data Storage

* JSON
* Metadata Files

---

## Repository Structure

```text
healthcare-genai-assistant/
│
├── raw/
├── cleaned/
├── structured_json/
├── metadata/
├── chunks/
├── notebooks/
├── docs/
│   ├── SOURCES.md
│   ├── ARCHITECTURE.md
│
├── scripts/
├── app/
└── README.md
```

## Safety Considerations

* No medical diagnosis
* No prescription recommendations
* No medicine dosage advice
* Emergency queries trigger safe healthcare guidance
* Source-backed responses wherever possible
* Fallback responses when reliable information is unavailable
