<div align="center">

# K-State

**Corrective RAG that doesn't trust its own retriever.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-Local_Index-0467DF?style=flat-square&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Inference-F55036?style=flat-square)](https://groq.com/)
[![Status](https://img.shields.io/badge/Status-Active_Development-yellow?style=flat-square)](https://github.com/GitTanish/K-STATE)

</div>

---

Most RAG systems are a straight line: retrieve, generate, ship. That works until it doesn't — and in production, it doesn't often enough to matter. K-State replaces that line with a loop. Retrieved chunks are scored before they reach the LLM. Bad scores trigger a web fallback. Everything gets filtered at the sentence level before generation. The system is skeptical of itself by design.

---

## The problem with "fetch-and-pray"

Standard RAG has a silent assumption baked in: that the retriever returns something useful. It doesn't validate this. It just retrieves and generates.

Three ways this breaks in practice:

- **Semantic drift** — The vector store returns chunks that are close in embedding space but wrong in meaning. The LLM gets plausible-sounding garbage and confidently hallucinates from it.
- **Stale knowledge** — A PDF you indexed months ago doesn't know about last week. The system has no mechanism to recognize this gap, let alone fill it.
- **Context poisoning** — A 1,000-token chunk gets retrieved because two sentences in it are relevant. The other 900 tokens are noise. The LLM reads all of it.

K-State treats each of these as an engineering problem, not an acceptable limitation.

---

## How it actually works

```
Query
  │
  ▼
FAISS retrieval  (k=4, BGE-Small-v1.5, CPU)
  │
  ▼
Chunk evaluator  (GPT-OSS 120B scores each chunk independently)
  │
  ├── any score > 0.7 ──────────────────────────► refine → generate
  │
  ├── scores between 0.3–0.7 ──► rewrite → web search ──► merge → refine → generate
  │
  └── all scores < 0.3 ────────► rewrite → web search ──────────► refine → generate
```

Every path — local, web, or merged — runs through the **Refiner** before generation. It splits context into individual sentences, scores each one for relevance, and discards the rest. The LLM only sees what passed the filter.

If refinement is too aggressive and strips everything, it falls back to the unfiltered sentences rather than returning empty. Graceful degradation over silent failure.

---

## What each piece does

**`retrieve`** — FAISS similarity search against the local index. Returns 4 chunks. Runs in milliseconds on CPU after the first-run index build.

**`eval_each_doc`** — Scores every retrieved chunk independently. This is where the verdict is decided:

| Verdict | Condition | Next step |
|---|---|---|
| `CORRECT` | At least one chunk scores > 0.7 | Refine locally |
| `AMBIGUOUS` | Some above 0.3, none above 0.7 | Merge local + web |
| `INCORRECT` | All chunks score < 0.3 | Abandon local, go web |

**`rewrite_query`** — Rewrites the original question into a tight keyword string. If the question implies recency ("latest," "this month," "recent"), it injects a temporal constraint — e.g., `last 30 days` — into the search query.

**`web_search`** — Hits DuckDuckGo structured API first. If rate-limited, falls back to the raw `DDGS` scraper automatically. No API key. If both return nothing, the system fails explicitly rather than hallucinating.

**`refine`** — The quality gate before generation. Decomposes context into sentences ≥ 25 chars, LLM-scores each for relevance, recomposes the survivors. Runs on every path.

**`generate`** — Synthesizes the final answer from refined context only. Returns "I don't know" if context is empty rather than inventing something.

---

## Graph

```
START → retrieve → eval_each_doc
                        │
              ┌─────────┴──────────┐
           CORRECT          AMBIGUOUS / INCORRECT
              │                     │
           refine         rewrite_query → web_search
              │                               │
           generate ◄──────────── refine ◄───┘
              │
             END
```

Nodes are pure functions. State is a typed dict. The graph is a DAG — no hidden side effects, no implicit memory between calls. Every decision is observable.

---

## Setup

```bash
git clone https://github.com/GitTanish/K-STATE.git
cd K-STATE
pip install -r requirements.txt
```

```bash
# .env
GROQ_API_KEY=your_key_here
```

Drop your PDFs into `books/`. Run:

```bash
python main.py
```

First run builds and persists the FAISS index — slow once. Every run after loads from disk in milliseconds. No re-embedding, no re-parsing.

---

## Configuration

| Variable | Default | What it controls |
|---|---|---|
| `UPPER_TH` | `0.7` | Score threshold for using local retrieval directly |
| `LOWER_TH` | `0.3` | Score below which local index is abandoned |
| `FAISS_PERSIST_DIR` | `./faiss_index` | Index save/load path |
| `k` | `4` | Chunks retrieved per query |

---

## Stack

| | |
|---|---|
| **Orchestration** | LangGraph |
| **Embeddings** | BAAI/BGE-Small-v1.5 — local, CPU |
| **Vector store** | FAISS — disk-persisted |
| **Evaluator / LLM** | GPT-OSS 120B via Groq |
| **Web search** | DuckDuckGo API + DDGS fallback |

---

## Known tradeoffs

**It's not fast.** On `INCORRECT` verdicts: 4 evaluation calls + rewrite + web search + N filter calls + generation. That's a lot of round trips. Groq inference keeps latency from being painful, but this isn't built for sub-second response times.

**The sentence filter can over-strip.** Dense technical content with lots of cross-referential sentences sometimes gets filtered too aggressively. The fallback to unfiltered context handles this, but the filter prompt is worth tuning for your domain.

**DuckDuckGo has no SLA.** The two-layer fallback handles most rate-limit cases. For anything with uptime requirements, swapping in Tavily or Serper at the `web_search` node is a one-function change.

---

## License

MIT. See [LICENSE](LICENSE).