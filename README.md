# Verifiable Local RAG (Offline Document QA & Fact-Checker)

An Offline, Hallucination-Free Retrieval-Augmented Generation (RAG) platform featuring Sentence-Level Fact-Checking, Deterministic Fallback Vector Engine, and Page-Aware Citation Verification. Built with Python, Streamlit, Microsoft Foundry Local SDK, and SQLite.

---

## Key Features

- Microsoft Foundry Local SDK Integration: Runs local LLMs (Qwen2.5-0.5B / Phi-4) fully offline without external API dependencies.
- Fault-Tolerant Deterministic Fallback Engine: Features a custom hash-based vector engine that ensures application continuity even if SDK runtime is absent.
- Zero-Hallucination Fact-Checker: Evaluates model responses sentence-by-sentence against source chunks using Jaccard word-overlap matching (0.0% - 100.0% confidence score).
- Query Relevance Filter: Prevents false positive citations by verifying meaningful keyword intersections.
- Markdown Table Parsing (pdfplumber): Extracts tabular data from PDFs as Markdown matrices to preserve numerical context.
- Dual-Language UI (TR / EN): Interactive Streamlit interface with instant language switching.

---

## System Architecture

```
[PDF / TXT Document]
        │
        ▼
[Parser & Table Extractor] ── (pdfplumber / Markdown Table Matrix)
        │
        ▼
[Chunker & Overlap Engine] ── (Page-Aware Metadata)
        │
        ▼
[Vector Embedding] ────────── (Microsoft Foundry SDK / Fallback Hash 384D)
        │
        ▼
[SQLite Vector Store] ─────── (JSON Serialized Vector Storage)
        │
        ▼
[Smart Retriever] ─────────── (Query Relevance Check + Cosine Similarity)
        │
        ▼
[Local LLM (Qwen2.5)] ─────── (Strict Zero-Hallucination System Prompt)
        │
        ▼
[Fact-Checker Verifier] ───── (Sentence-Level Citation Matching & Scoring)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher.
- pip package manager.

### 2. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/blosny/verifiable-local-rag.git
cd verifiable-local-rag
pip install -r requirements.txt
```

### 3. Run Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The web UI will open automatically at http://localhost:8501.

---

## Testing & Verification

Run the automated offline test suite covering target document retrieval, table extraction, and hallucination checks:

```bash
python -m notes.test_suite
```

---

## Project Structure

```
verifiable-local-rag/
├── app.py                  # Streamlit Web Application (i18n TR/EN UI)
├── src/
│   ├── database.py         # SQLite Vector Store & Cosine Similarity
│   ├── ingest.py           # PDF/TXT Parser & Markdown Table Extractor
│   ├── retriever.py        # Smart Retriever & Query Relevance Filter
│   ├── llm.py              # Microsoft Foundry Local SDK Client & Fallback Engine
│   └── verifier.py         # Sentence-Level Jaccard Fact-Checker
├── data/                   # Sample PDF Documents
├── notes/                  # Test Suites & Engineering Decision Logs
└── requirements.txt        # Python Dependencies
```

---

## License

Distributed under the MIT License.
