# TelecomCo Support Assistant

A Retrieval-Augmented Generation (RAG) system for telecom customer support. It answers natural language questions by intelligently routing between two pipelines: **Hybrid Vector RAG** for knowledge-based queries and **SQL RAG** for analytical/statistical queries.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                         │
│                                                             │
│   tickets.db (SQLite)   telecom_guide.pdf    faq.csv        │
│   20 support tickets    Policy/guide doc     Q&A pairs      │
└────────┬────────────────────────┬────────────┬─────────────┘
         │                        │            │
         ▼                        ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION  (src/ingest.py)              │
│                                                             │
│  load_tickets()        load_pdf()          load_csv()       │
│  Each ticket →         Docling parser →    pandas →         │
│  1 Document            HybridChunker       1 doc/row        │
│  (no chunking)         (128 token chunks)  (no chunking)    │
└───────────────────────────────┬─────────────────────────────┘
                                │  all_docs (combined list)
                                ▼
┌─────────────────────────────────────────────────────────────┐
│               HYBRID VECTORSTORE  (Qdrant)                  │
│                                                             │
│   Dense embeddings: sentence-transformers/all-MiniLM-L6-v2  │
│   Sparse embeddings: Prithivida/Splade_PP_en_v1 (SPLADE)   │
│   Retrieval mode: HYBRID (dense + sparse combined)          │
└───────────────────────────────┬─────────────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         │         QUERY ROUTER  (src/app.py)           │
         │   is_sql_question() — keyword detection      │
         └──────────┬───────────────────────────────────┘
                    │
        ┌───────────┴──────────────┐
        │ analytical question?     │ knowledge question?
        ▼                          ▼
┌───────────────┐        ┌──────────────────────────────────┐
│  SQL RAG      │        │  VECTOR RAG  (src/chain.py)      │
│ (sql_chain.py)│        │                                  │
│               │        │  Basic mode:                     │
│ 1. LLM writes │        │    retrieve top-5 → stuff → LLM  │
│    SQL query  │        │                                  │
│ 2. Run query  │        │  Reranking mode:                 │
│    on tickets │        │    retrieve top-10 candidates    │
│    .db        │        │    → cross-encoder reranker      │
│ 3. LLM turns  │        │      (ms-marco-MiniLM-L-6-v2)   │
│    result to  │        │    → keep top-3 → LLM            │
│    plain text │        └──────────────┬───────────────────┘
└───────┬───────┘                       │
        └──────────────┬────────────────┘
                       ▼
              LLM: Groq (gpt-oss-20b)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  GRADIO UI  (src/app.py)                    │
│                                                             │
│  Chat interface · Source citations · Chain label display    │
│  Toggle: enable/disable reranking                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Two RAG Pipelines

### 1. Hybrid Vector RAG

Used for questions about policies, troubleshooting steps, and support ticket resolutions.

- **Retrieval**: Qdrant hybrid search combines dense (semantic) and sparse (keyword/SPLADE) vectors for better recall than either alone.
- **Basic mode**: fetches top-5 documents, stuffs them into the prompt context, and sends to the LLM.
- **Reranking mode**: fetches top-10 candidates, applies a cross-encoder reranker to score each document against the query, keeps the top-3 most relevant, then sends to the LLM. Slower but more precise.

### 2. SQL RAG

Used for analytical questions (counts, breakdowns, statistics).

- The LLM generates a SQL query from the natural language question.
- The query runs directly against `tickets.db`.
- The raw SQL result is passed back to the LLM to produce a clean natural language answer.

**Routing keywords** that trigger SQL RAG: `how many`, `count`, `total`, `most common`, `breakdown`, `resolved vs`, `escalated`, `average`, `statistics`, etc.

---

## Data Sources

| Source | Format | Loading strategy | Chunking |
|---|---|---|---|
| `tickets.db` | SQLite | `sqlite3` → LangChain `Document` | None — each ticket is a complete unit |
| `telecom_guide.pdf` | PDF | Docling `DocumentConverter` | `HybridChunker`, 128 tokens, heading context preserved |
| `faq.csv` | CSV | `pandas` → LangChain `Document` | None — each row is a complete Q&A |

---

## Project Structure

```
telecom-rag/
├── src/
│   ├── ingest.py       # Data loading and vectorstore construction
│   ├── chain.py        # Vector RAG chains (basic + reranking)
│   ├── sql_chain.py    # SQL RAG chain and query router
│   └── app.py          # Gradio UI and request orchestration
├── data/
│   └── tickets.db      # SQLite support tickets database
├── main.py             # CLI entrypoint for testing all chains
└── pyproject.toml
```

---

## Setup

**Prerequisites**: Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Set your Groq API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

---

## Running

**Gradio UI:**
```bash
uv run python src/app.py
```

**CLI (tests all chains):**
```bash
uv run python main.py
```

---

## Example Questions

| Question | Pipeline used |
|---|---|
| "What should I do about unexpected roaming charges?" | Vector RAG |
| "My phone shows full bars but can't load websites" | Vector RAG |
| "How do I fix my APN settings?" | Vector RAG |
| "How many tickets are in each category?" | SQL RAG |
| "How many tickets have been resolved vs escalated?" | SQL RAG |
| "What is the most common connectivity issue?" | SQL RAG |
