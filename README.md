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
> A dedicated node that scores retrieved document relevance and bifurcates the logic flow based on the **Quality of Truth**:
>
> | Verdict | Action |
> |---|---|
> | **Correct** | Refines context → generates answer |
> | **Ambiguous** | Triggers hybrid merge of local data + targeted web-search fallback |
> | **Incorrect** | Rejects local index entirely → activates query-rewriting search agent |

> **Corrective Loop** — *The Safety Net*
>
> When local retrieval is insufficient, the system autonomously rewrites the query for web-scale search (Tavily), fetches external context, and injects it back into the generation node.

---

## Architecture & Design Decisions

### 1. Matryoshka-Optimized Semantic Caching

K-State uses **Nomic-v1.5** with Matryoshka embeddings, enabling vector truncation from 768 → 256 dimensions without significant recall loss. The result: a semantic cache that is faster and uses a fraction of the memory footprint of standard implementations.

### 2. Local-First Embedding Runtime

All embeddings run through local `sentence-transformers` — no external API calls. The "Knowledge Translation" step happens entirely on local compute before any data hits the inference cloud. This eliminates latency and sidesteps privacy risks.

### 3. State-Machine Orchestration

K-State is an **Actor-Model** style DAG, not a linear script. Each component — Evaluation, Rewriting, Searching, Refinement — is a modular node. This enables granular auditing of the system's decision process: transparent engineering-grade tooling, not a black box.

### 4. Context Refinement

Before final generation, a **Decomposition & Filtering** node breaks retrieved documents into individual "knowledge strips," removes irrelevant filler, and feeds only high-density facts to the LLM. This minimizes distractors and ensures strictly grounded output.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph / LangChain |
| **Local Embeddings** | Nomic-Embed-v1.5 (via Sentence-Transformers) |
| **Vector Engine** | FAISS-CPU (Cache) & ChromaDB (Index) |
| **Inference** | Groq |
| **Search Agent** | DuckDuckGo AI API |

---

## Design Philosophy

K-State is built for environments where **accuracy is a mechanical necessity**, not a side effect. It proves that the solution to LLM unreliability isn't more parameters — it's better **state management**.

---

<div align="center">
<sub>Built as a demonstration that retrieval systems should think, not just fetch.</sub>
</div>