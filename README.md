# verifiable-local-rag

> A modular, self-correcting offline Local RAG system with SQLite vector storage, page-aware chunking, and source citation verification built with Microsoft Foundry Local SDK.

---

## 🏛️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 1. UI / CLIENT                                    |
|                       (Streamlit - Web Chat & Citation View)                       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              2. INGESTION PIPELINE                                |
|  - PDF / TXT Loader & Parser                                                      |
|  - Text Chunker (Page-Aware Metadata)                                             |
|  - Embedding Generator (Foundry Local SDK / Vector Engine)                        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               3. VECTOR STORE LAYER                               |
|  - SQLite DB (Content + Metadata + Vector Embeddings)                             |
|  - Cosine Similarity Retriever                                                    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        4. LLM & CITATION VERIFIER ENGINE                          |
|  - Foundry Local LLM Runtime (Qwen / Phi)                                         |
|  - Citation Matcher (Sentence-level Source Matching & Verification Score)          |
+-----------------------------------------------------------------------------------+
```
