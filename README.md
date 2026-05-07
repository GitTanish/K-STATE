<div align="center">

# K-State

### Agentic Knowledge Runtime with Self-Correcting Retrieval

[![Status](https://img.shields.io/badge/Status-In_Development-yellow?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=flat-square&logo=langchain&logoColor=white)]()
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Cache-0467DF?style=flat-square&logo=meta&logoColor=white)]()

---

**K-State** is a high-fidelity knowledge runtime that replaces the traditional linear RAG pipeline with a **self-correcting state machine**. It treats retrieval as a fallible process requiring validation and memory — not a single deterministic step you blindly trust.

</div>

---

## The Problem

Most RAG implementations follow a deterministic path:

```
Query → Embed → Search → Generate
```

This assumes the retriever is always right. In production, it isn't. Two failure modes dominate:

| Failure Mode | What Happens |
|---|---|
| **Low-Fidelity Retrieval** | The vector DB returns semantically similar but factually irrelevant noise — the LLM hallucinates from bad context. |
| **Computational Redundancy** | Identical or equivalent queries re-run the full retrieval/generation loop, wasting GPU cycles on already-solved problems. |

---

## The Solution: Agentic Knowledge Orchestration

K-State introduces a **Decision Layer** built on **LangGraph** that transforms retrieval into a modular state machine capable of evaluating, rejecting, and re-searching autonomously.

### Core Components

> **Semantic Cache** — *The Pre-processor*
>
> A high-speed similarity layer backed by a local FAISS index. Performs sub-millisecond lookups to check if a grounded answer already exists in local state, bypassing the LLM entirely for frequent queries.

> **Knowledge Evaluator** — *The Quality Gate*
>
> A dedicated node that evaluates sentence relevance and keeps only the strips that directly answer the query.

> **Refined Generation** — *The Grounding Step*
>
> The final answer is generated only from the filtered strips, reducing noise and hallucinations.

---

## Architecture & Design Decisions

### 1. Local-First Embedding Runtime

All embeddings run locally using **BAAI/bge-small-en-v1.5** via `langchain-huggingface`. This keeps retrieval fast and private while remaining high-quality for small-to-medium corpora.

### 2. State-Machine Orchestration

K-State is a **LangGraph** DAG, not a linear script. Each component — Retrieval, Refinement, Generation — is a modular node, enabling transparent and auditable control flow.

### 3. Context Refinement

Before final generation, a **Decomposition & Filtering** step breaks retrieved documents into individual "knowledge strips," removes irrelevant filler, and feeds only high-density facts to the LLM.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph / LangChain |
| **Local Embeddings** | BAAI/bge-small-en-v1.5 (via langchain-huggingface) |
| **Vector Engine** | FAISS-CPU |
| **Inference** | Groq |

---

## Design Philosophy

K-State is built for environments where **accuracy is a mechanical necessity**, not a side effect. It proves that the solution to LLM unreliability isn't more parameters — it's better **state management**.

---

<div align="center">
<sub>Built as a demonstration that retrieval systems should think, not just fetch.</sub>
</div>