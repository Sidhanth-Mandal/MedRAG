# 🧠 MedRAG — Deep Project Explainer

> This document explains the **complete internal working** of MedRAG — every module, every node, every database operation, and every data transformation, from a user typing a question to receiving a cited medical answer.

---

## Table of Contents

1. [Big Picture — The Two Phases](#1-big-picture--the-two-phases)
2. [Phase 1 — Data Pipeline (Offline)](#2-phase-1--data-pipeline-offline)
   - [Ingestion](#21-ingestion)
   - [Chunking](#22-chunking)
   - [Embedding](#23-embedding)
   - [Qdrant Upload](#24-qdrant-upload)
   - [BM25 Index](#25-bm25-index)
3. [Phase 2 — The Agentic RAG (Online, Per Query)](#3-phase-2--the-agentic-rag-online-per-query)
   - [AgentState — The Shared Data Envelope](#31-agentstate--the-shared-data-envelope)
   - [LangGraph Graph Structure](#32-langgraph-graph-structure)
   - [Node Deep Dives](#33-node-deep-dives)
     - [summarizer_node](#331-summarizer_node)
     - [intake_node](#332-intake_node)
     - [direct_answer_node](#333-direct_answer_node)
     - [safety_node](#334-safety_node)
     - [router_node](#335-router_node)
     - [query_rewriter_node](#336-query_rewriter_node)
     - [retriever_node (single-query)](#337-retriever_node-single-query)
     - [multi_retriever_node (multi-query)](#338-multi_retriever_node-multi-query)
     - [grader_node](#339-grader_node)
     - [contradiction_node](#3310-contradiction_node)
     - [answer_node](#3311-answer_node)
4. [The Retrieval Engine](#4-the-retrieval-engine)
   - [Dense Retrieval — Qdrant](#41-dense-retrieval--qdrant)
   - [Sparse Retrieval — BM25](#42-sparse-retrieval--bm25)
   - [Hybrid Fusion — RRF](#43-hybrid-fusion--rrf)
   - [Reranking — Cohere](#44-reranking--cohere)
5. [The LLM Layer — Groq with Key Rotation](#5-the-llm-layer--groq-with-key-rotation)
6. [The Database Layer — Supabase / PostgreSQL](#6-the-database-layer--supabase--postgresql)
   - [Schema](#61-schema)
   - [All DB Operations](#62-all-db-operations)
   - [Chat History Strategy (Summary + Recent)](#63-chat-history-strategy-summary--recent)
   - [LangGraph Checkpointer](#64-langgraph-checkpointer)
7. [The API Layer — FastAPI](#7-the-api-layer--fastapi)
   - [Authentication Flow](#71-authentication-flow)
   - [Chat Endpoint Flow](#72-chat-endpoint-flow)
   - [Session Management Endpoints](#73-session-management-endpoints)
   - [Pydantic Models](#74-pydantic-models)
8. [Frontend — How the UI Works](#8-frontend--how-the-ui-works)
9. [Configuration System](#9-configuration-system)
10. [End-to-End Request Walkthrough](#10-end-to-end-request-walkthrough)

---

## 1. Big Picture — The Two Phases

MedRAG operates in two completely separate phases:

```
PHASE 1 (Offline — run once)
─────────────────────────────────────────────────────────────────
  PubMed API  ──┐
                ├──► Raw JSON ──► Chunker ──► Embedder ──► Qdrant Cloud
  MedlinePlus ──┘                       └──────────────────► BM25 .pkl

PHASE 2 (Online — runs on every user message)
─────────────────────────────────────────────────────────────────
  User Message
      │
  FastAPI /api/chat
      │
  LangGraph Agent (10-node state machine)
      │
  ┌─── Hybrid Retrieval (BM25 + Qdrant + RRF + Cohere Rerank)
  │
  └─── Groq LLaMA 3.3 70B (answer with citations)
      │
  Supabase (persist message + citations)
      │
  JSON Response → Frontend
```

**Phase 1** only needs to run once (or whenever new documents are added).  
**Phase 2** runs in milliseconds-to-seconds on every chat message.

---

## 2. Phase 1 — Data Pipeline (Offline)

### 2.1 Ingestion

**Entry point:** `python -m ingestion.run_ingestion`

Two fetchers run in sequence:

#### `ingestion/pubmed_fetcher.py`

Uses **Biopython's Entrez API** to query NCBI PubMed:

1. Calls `Entrez.esearch()` with a condition-specific query (e.g. `"type 2 diabetes mellitus management treatment"`), a date range (`2019/01/01` → `2025/12/31`), and `sort="relevance"`. Returns up to 350 PMIDs per condition.
2. Calls `Entrez.efetch()` in batches of 50, returning XML. Respects NCBI's rate limit with a 350ms sleep between batches.
3. Parses each XML article: extracts PMID, title, abstract text, authors, journal name, and publication year.
4. Skips articles with no abstract or abstracts shorter than 50 characters.
5. Tags each record with `source_type: "research"` and `condition: "<slug>"`.

**Output:** `data/raw/pubmed_abstracts.json` — a flat JSON array of document dicts.

**Conditions fetched:**
| Condition Slug | PubMed Search Query |
|---|---|
| `type2_diabetes` | `"type 2 diabetes mellitus management treatment"` |
| `hypertension` | `"hypertension treatment blood pressure management"` |
| `asthma` | `"asthma treatment management inhaler"` |

#### `ingestion/medlineplus_fetcher.py`

Scrapes **MedlinePlus** health topic pages using the MedlinePlus Connect API and BeautifulSoup HTML parsing. Extracts patient-facing guidelines tagged with `source_type: "guideline"`.

**Output:** `data/raw/medlineplus_guidelines.json`

---

### 2.2 Chunking

**File:** `pipeline/chunker.py`

Raw documents are split into **overlapping token-aware chunks** using `tiktoken` (the `cl100k_base` tokenizer):

```
chunk_size    = 400 tokens
chunk_overlap = 50 tokens
```

**Algorithm:**
1. Encode the full document text to a token list.
2. Slice a window of `chunk_size` tokens starting at `start`.
3. If not at the end, try to trim the chunk to the last sentence boundary (looks for `.`, `!`, or `?` followed by whitespace within the last 200 tokens) to avoid mid-sentence cuts.
4. Advance `start` by `chunk_size - chunk_overlap = 350` tokens.
5. Repeat until the full document is covered.

Each output chunk dict carries **all original document metadata** plus:
- `chunk_index` — position within the document
- `chunk_text` — the actual text of this chunk
- `chunk_id` — deterministic ID: `"{doc_id}_chunk_{chunk_index}"`
- `token_count` — number of tokens in this chunk

**Why overlapping chunks?** The 50-token overlap ensures that a sentence spanning a chunk boundary doesn't get split across two chunks with no context on either side.

---

### 2.3 Embedding

**File:** `pipeline/embedder.py`

The `CloudflareEmbedder` class calls the **Cloudflare Workers AI REST API** with model `@cf/baai/bge-base-en-v1.5`:

- Processes chunks in **batches of 64** (configurable).
- POSTs `{"text": [...batch of strings...]}` to `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`.
- Handles **rate limiting (HTTP 429)** with exponential backoff: `2^attempt × 2.0` seconds.
- Returns `response["result"]["data"]` — a list of 768-dimensional float vectors.

**Why Cloudflare?** bge-base-en-v1.5 is a strong open embedding model, and Cloudflare Workers AI provides it for free at reasonable limits, making it cost-effective for a personal project.

---

### 2.4 Qdrant Upload

**File:** `pipeline/qdrant_uploader.py`

1. **`ensure_collection()`** — Creates the Qdrant collection if it doesn't exist:
   - Vector dimension: **768** (matching bge-base-en-v1.5)
   - Distance metric: **Cosine**
   - Payload indexes created on `source_type`, `condition`, and `doc_id` (keyword indexes for O(1) filtered search)

2. **`upsert_chunks()`** — Uploads in batches of 100:
   - Converts `chunk_id` strings to **UUID v5** (deterministic — same chunk always gets the same UUID, enabling idempotent re-uploads).
   - Stores the full chunk metadata as a **Qdrant payload** alongside the vector.
   - Uses `client.upsert(..., wait=True)` to ensure writes are confirmed before continuing.

---

### 2.5 BM25 Index

**File:** `retrieval/bm25_retriever.py`

Built using `rank-bm25` (BM25Okapi algorithm):

1. **Tokenization:** lowercases text, strips non-alphanumeric characters, splits on whitespace.
2. **Indexing:** `BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)` — standard Okapi BM25 parameters.
3. **Serialization:** The `(bm25_model, chunks_list)` pair is pickled to `data/bm25_index.pkl` for fast reload on subsequent runs.

**Important:** The BM25 index is built directly from the **raw JSON files** (not from Qdrant), so it works independently and fully offline. At query time, the index is loaded once and cached as a module-level singleton.

**Why BM25 + Dense (Hybrid)?** Dense retrieval excels at semantic similarity but can miss exact keyword matches. BM25 is the opposite — great at exact terms, but blind to synonyms. Combining both with RRF captures the strengths of each.

---

## 3. Phase 2 — The Agentic RAG (Online, Per Query)

### 3.1 AgentState — The Shared Data Envelope

**File:** `agent/state.py`

`AgentState` is a `TypedDict` — the **single shared data structure** that flows through every node in the LangGraph graph. Nodes read from it and return partial dicts that get merged back in.

| Field | Type | Set by | Description |
|---|---|---|---|
| `query` | `str` | API caller | The raw user question |
| `session_id` | `str` | API caller | Thread/session identifier |
| `messages` | `list` (add_messages) | answer/direct_answer nodes | LangGraph message accumulator |
| `chat_summary` | `str` | summarizer_node | Rolling LLM summary of older history |
| `history_context` | `str` | `run_agent()` | Formatted summary + recent messages for prompts |
| `needs_rag` | `bool` | intake_node | False = skip retrieval |
| `route` | `str` | router_node | `'research'` / `'guideline'` / `'both'` |
| `retrieved_docs` | `list[dict]` | retriever/multi_retriever nodes | Top reranked chunks |
| `search_query` | `str` | query_rewriter_node | Primary search query (also rewritten by grader) |
| `search_queries` | `list[str]` | query_rewriter_node | All sub-queries (1=single, 2-3=multi) |
| `rewrite_count` | `int` | grader_node | How many rewrites attempted (max 2) |
| `docs_relevant` | `bool` | grader_node | Did the grader accept the docs? |
| `is_safe` | `bool` | safety_node | False = refuse |
| `is_refused` | `bool` | safety_node | Final refused flag |
| `refuse_reason` | `str` | safety_node | Reason string |
| `has_contradiction` | `bool` | contradiction_node | Were conflicts found? |
| `contradiction_details` | `str` | contradiction_node | Human-readable conflict description |
| `answer` | `str` | answer/direct_answer nodes | Final answer text |
| `citations` | `list[dict]` | answer_node | Structured citation objects |

The `messages` field uses LangGraph's `add_messages` **reducer** — when a node returns new messages, they are **appended** to the existing list rather than replacing it. This is critical for LangGraph's built-in message accumulation.

---

### 3.2 LangGraph Graph Structure

**File:** `agent/graph.py`

The graph is assembled using `StateGraph(AgentState)`:

```
START
  │
  ▼
[summarize]  ──────────────────────────────────────────────────────────────────┐
  │                                                                             │
  ▼                                                                             │
[intake]  ── needs_rag=False ──► [direct_answer] ──────────────────────► END   │
  │                                                                             │
  └── needs_rag=True ──► [safety]                                              │
                            │                                                   │
                    is_safe=False ──────────────────────────────────────► END   │
                            │                                                   │
                    is_safe=True ──► [router] ──► [query_rewrite]               │
                                                       │                        │
                                          len(queries)==1 ──► [retrieve]        │
                                          len(queries) >1 ──► [multi_retrieve]  │
                                                       │                        │
                                                    [grade]                     │
                                                  /        \                    │
                                           relevant=True  relevant=False        │
                                               │          AND retries<2         │
                                               │              │                 │
                                               │          [retrieve] (retry)    │
                                               ▼                                │
                                         [contradiction]                        │
                                               │                                │
                                           [answer] ──────────────────────► END │
                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Conditional edge functions** in `graph.py` implement the branching logic:

| Edge Function | Reads | Returns |
|---|---|---|
| `after_intake` | `state["needs_rag"]` | `"rag"` or `"conversational"` |
| `after_safety` | `state["is_safe"]` | `"safe"` or `"refused"` |
| `after_query_rewrite` | `len(state["search_queries"])` | `"single"` or `"multi"` |
| `after_grade` | `state["docs_relevant"]`, `state["rewrite_count"]` | `"retry"` or `"proceed"` |

The graph is **compiled with a PostgreSQL checkpointer** (backed by Supabase) when `SUPABASE_DB_URL` is available. The checkpointer stores the full `AgentState` after each invocation, keyed by `session_id` (called `thread_id` in LangGraph). This is what enables the `messages` list to accumulate across turns for the same session.

---

### 3.3 Node Deep Dives

#### 3.3.1 `summarizer_node`
**File:** `agent/nodes/summarizer.py`

**Trigger condition:** Runs at the **start of every turn**.

**Logic:**
1. Queries `db.count_messages(session_id)` to get the total message count for this session.
2. If count < `cfg.agent.chat_history_window` (default: **10**), returns immediately with the existing `chat_summary` from state (no-op).
3. If count ≥ 10, fetches the **full conversation history** from Postgres.
4. Formats it as `"User: ...\nAssistant: ..."` pairs, truncating each message to 300 chars.
5. Calls the LLM with a summarization prompt asking for a **max 150-word rolling summary** covering: medical topics discussed, key facts/answers, and unresolved questions.
6. Saves the new summary to `session_summaries` table via `db.save_summary()`.
7. Returns `{"chat_summary": summary}` to update the state.

**Why rolling summarization?** Rather than truncating history or sending an ever-growing context window to the LLM (expensive and eventually hitting token limits), the system compresses old turns into a concise summary. The agent's prompts then consume `summary + last N raw messages` instead of the full history.

---

#### 3.3.2 `intake_node`
**File:** `agent/nodes/intake.py`

**Purpose:** Acts as a **gatekeeper** to prevent the expensive retrieval pipeline from firing on conversational turns like "hi", "thanks", "what did you just say?".

**Logic:**
1. Calls the LLM with a classification prompt, providing the user's `query` and the `history_context`.
2. Prompt instructs the LLM to respond with **exactly one word: `RAG` or `CHAT`**.
3. Uses `temperature=0.0` and `max_tokens=5` — a single-token classification, the cheapest possible LLM call.
4. Defaults to `RAG` on ambiguous output or on exception (safe default: never miss a medical question).

**Classification examples the prompt encodes:**
- `"hi there"` → CHAT
- `"what is type 2 diabetes?"` → RAG
- `"what did you say about metformin?"` → CHAT (answerable from history)
- `"tell me more"` → RAG (when prior context was medical)

**State update:** Sets `needs_rag: bool`.

---

#### 3.3.3 `direct_answer_node`
**File:** `agent/nodes/direct_answer.py`

**When reached:** Only when `intake_node` returned `needs_rag=False`.

**Logic:**  
Calls the LLM with a conversational prompt that includes the full `history_context`. Uses `temperature=0.4` for a slightly more natural, friendly tone (unlike the factual nodes which use `0.0–0.2`).

Explicitly instructs the LLM: **do not make up medical information** — if the question actually needs retrieval, say so.

**State updates:** Sets `answer`, `citations: []`, `is_refused: False`, and appends `HumanMessage + AIMessage` to `messages`.

---

#### 3.3.4 `safety_node`
**File:** `agent/nodes/safety.py`

**Purpose:** Detects and refuses **personalized medical advice requests** (diagnosis, dosing, "should I take X?").

**Two-stage approach:**

**Stage 1 — Keyword fast-path (no LLM):**  
Runs 13 pre-compiled regex patterns against the query. Examples:
- `r"\bdiagnose me\b"` — catches "can you diagnose me?"
- `r"\bshould i (take|stop|start)\b"` — catches "should I stop taking metformin?"
- `r"\bmy (blood sugar|blood pressure)\b"` — catches "my blood pressure is 160/100"
- `r"\bam i (sick|diabetic)\b"` — catches "am I diabetic?"

If any pattern matches → immediately returns the refusal (no LLM call needed).

**Stage 2 — LLM classifier (ambiguous cases):**  
If no keyword matched, calls the LLM with a safety classification prompt, `temperature=0.0`, `max_tokens=5`. The LLM returns `SAFE` or `UNSAFE`.

If LLM call fails for any reason → defaults to **SAFE** (never block a legitimate query due to an API failure).

**Refusal message:** A fixed, empathetic message that says the system provides general educational information only and directs the user to a qualified healthcare professional.

**State updates when refusing:** Sets `is_safe: False`, `is_refused: True`, `refuse_reason: "personalized_medical_advice"`, and pre-populates `answer` with the refusal text.

---

#### 3.3.5 `router_node`
**File:** `agent/nodes/router.py`

**Purpose:** Decides **which knowledge source to search** — PubMed research abstracts, MedlinePlus guidelines, or both.

**Logic:** Single LLM call with `temperature=0.0`, `max_tokens=10`. The prompt gives concrete examples:
- `"What does the research say about metformin?"` → `research`
- `"What lifestyle changes help with high blood pressure?"` → `guideline`
- `"What is asthma?"` → `both`

**Default fallback:** If LLM fails or returns an unexpected value → `"both"` (safe, never misses results).

**State update:** Sets `route: str`. This value is passed to the retriever as a `source_type` filter on Qdrant and the BM25 results.

---

#### 3.3.6 `query_rewriter_node`
**File:** `agent/nodes/query_rewriter.py`

**Two jobs in one LLM call:**

**Job 1 — Context resolution:**
Resolves pronouns and vague references using `history_context`. Examples:
- User: `"side effects?"` after discussing metformin → `"metformin side effects type 2 diabetes"`
- User: `"the second one"` → resolves to the actual drug mentioned

**Job 2 — Multi-query decomposition:**
If the question is complex (comparison, multi-condition, multi-drug), the LLM emits **2–3 sub-queries**:
- `"compare metformin and SGLT2 inhibitors"` → two sub-queries, one per drug

**Output parsing:** The LLM is prompted to return a numbered `QUERIES:` block. The `_parse_queries()` function parses it with a regex looking for lines starting with `\d+[.)]`. Hard-capped at 3 sub-queries. Falls back to the original query if parsing fails.

**State updates:** Sets `search_queries: list[str]` and `search_query: str` (the first query — used as the grader's rewrite target).

**Why this matters:** Without context resolution, follow-up questions like "tell me more about that drug" would search for "that drug" in the vector store, yielding nothing. Without decomposition, "compare X and Y" gets a single combined search that may retrieve docs about only one of them.

---

#### 3.3.7 `retriever_node` (single-query)
**File:** `agent/nodes/retriever.py`

**When used:** When `len(search_queries) == 1`.

**Logic:**
1. Reads `search_query` from state (may be a grader-rewritten version).
2. Maps `route` → `source_type` filter (`None` when `route == "both"`).
3. Calls `hybrid_retriever.search()` — returns up to `rerank_top_k` (30) candidates.
4. Calls `reranker.rerank()` — returns the final top `final_top_k` (5) chunks.

**State update:** Sets `retrieved_docs: list[dict]`.

---

#### 3.3.8 `multi_retriever_node` (multi-query)
**File:** `agent/nodes/multi_retriever.py`

**When used:** When `len(search_queries) > 1`.

**Logic:**
1. Runs `hybrid_retriever.search()` for **each sub-query independently**.
2. Merges all candidate pools into a single list.
3. **Deduplicates** by `chunk_id` — if the same chunk appeared in multiple sub-query results, keeps the copy with the highest `rerank_score`.
4. Reranks the merged pool using the **original user query** (not the sub-queries) as the rerank anchor. This ensures the final ordering is holistic, relevant to the user's full intent.

**Why rerank against the original query?** Each sub-query is optimized for one angle of the question. The final reranking step with the full original question selects the best overall 5 chunks rather than the best per sub-query.

---

#### 3.3.9 `grader_node`
**File:** `agent/nodes/grader.py`

**Purpose:** Self-RAG style quality check — if retrieved docs are irrelevant, rewrite the query and retry.

**Logic:**
1. Reads `retrieved_docs` and `rewrite_count` from state.
2. If no docs were retrieved → trigger rewrite (if budget allows).
3. Formats the top-5 chunks (first 300 chars each) and asks the LLM: **"Do these excerpts contain information directly relevant to answering the question? yes | no"**
4. Uses `temperature=0.0`, `max_tokens=5` — single-token verdict.
5. If verdict is `"no"` and `rewrite_count < max_rewrite_retries` (2):
   - Calls `_rewrite_query()` — a separate LLM call that asks the LLM to improve the search query given that the previous one returned irrelevant results.
   - Returns `docs_relevant: False` and the new `search_query`.
   - The graph's `after_grade` edge returns `"retry"` → loops back to the `retrieve` node.
6. If relevant OR retries exhausted → returns `docs_relevant: True/False` and proceeds.

**Key design:** The grader only loops up to **2 times** (`max_rewrite_retries`). Even if docs are ultimately not relevant, the answer node still runs — it falls back to a "could not find relevant information" message rather than crashing.

---

#### 3.3.10 `contradiction_node`
**File:** `agent/nodes/contradiction.py`

**Purpose:** Detects when retrieved PubMed research contradicts MedlinePlus guidelines (or different research papers contradict each other).

**Logic:**
1. Requires at least 2 retrieved docs to run (otherwise trivially no conflict).
2. Formats up to 6 docs with their source type and title as a labelled list.
3. Asks the LLM to look for: conflicting treatment recommendations, opposite findings on drug efficacy/safety, disagreement on targets (e.g. blood pressure goals).
4. LLM responds with either `NO_CONFLICT` or `YES: <1-2 sentence description>`.

**Example contradiction:** Research paper says "intensive glycemic control reduces cardiovascular events" while a guideline says "aggressive targets increase risk of hypoglycemia — individualize targets". Both are true but apparently contradictory — the contradiction node flags this, and the answer node explicitly acknowledges the disagreement rather than silently picking one view.

**State updates:** Sets `has_contradiction: bool` and `contradiction_details: str`.

---

#### 3.3.11 `answer_node`
**File:** `agent/nodes/answer.py`

**Purpose:** Generates the final answer with inline citations.

**Logic:**
1. Formats retrieved docs as numbered source blocks: `[1] (RESEARCH) Title\nChunk text...`
2. If `has_contradiction=True`, appends a special `conflict_instruction` to the prompt telling the LLM to **explicitly acknowledge the disagreement** rather than silently picking one view.
3. If `history_context` exists (follow-up question), prepends it to the prompt so the LLM understands what `"it"` or `"that drug"` refers to.
4. Instructs the LLM to cite sources inline as `[1]`, `[2]`, etc. and to **stay within 300 words**.
5. Builds structured `citations` dicts — includes index, title, source_type, URL, PMID, pub_date, a 200-char snippet, and the Cohere rerank score.
6. For PubMed sources without a pre-filled URL, constructs the URL as `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`.
7. Appends `HumanMessage + AIMessage` to `messages` for LangGraph's checkpointer.

**State updates:** Sets `answer`, `citations`, and `messages`.

---

## 4. The Retrieval Engine

### 4.1 Dense Retrieval — Qdrant

**File:** `retrieval/dense_retriever.py`

1. Calls `embedder.embed_one(query)` → 768-dim vector via Cloudflare Workers AI.
2. Calls `qdrant_client.query_points()` (Qdrant SDK ≥1.9 API) with:
   - `query=query_vector` — the dense query embedding
   - `query_filter` — optional `FieldCondition` filters on `source_type` and `condition`
   - `limit=dense_top_k` (30 by default)
   - `with_payload=True` — returns the full chunk metadata stored as the payload
3. Returns a list of chunk dicts with `dense_score` field added.

Qdrant uses **cosine similarity** (configured at collection creation time). The collection has keyword payload indexes on `source_type` and `condition`, making filtered queries as fast as unfiltered ones.

---

### 4.2 Sparse Retrieval — BM25

**File:** `retrieval/bm25_retriever.py`

1. Loads the pickled index from `data/bm25_index.pkl` (cached in memory as a module-level singleton after the first load).
2. Tokenizes the query: lowercase + strip punctuation + split on whitespace.
3. `bm25.get_scores(tokenized_query)` returns a score for every chunk in the corpus.
4. Filters by `source_type` and/or `condition` if specified.
5. Sorts by score descending, returns top `bm25.top_k` (30) chunks with `bm25_score` field.

BM25Okapi with `k1=1.5, b=0.75` are standard values. `k1` controls term frequency saturation (higher = more weight on rare repeated terms). `b=0.75` normalizes for document length.

---

### 4.3 Hybrid Fusion — RRF

**File:** `retrieval/hybrid_retriever.py`

**Reciprocal Rank Fusion** combines the BM25 and dense ranking lists:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d) + 1}$$

Where $k = 60$ (configurable via `cfg.retrieval.rrf_k`). The $+1$ adjusts for 0-indexed ranks.

**Implementation:**
1. Build a dict keyed by `chunk_id`.
2. For each chunk in BM25 results (ranked 0 → N): add `1 / (60 + rank + 1)` to its `rrf_score`.
3. Repeat for dense results.
4. If a chunk appears in both lists, it accumulates scores from both — this is the key benefit. A chunk ranked #5 in BM25 and #5 in dense scores higher than a chunk ranked #1 in only one list.
5. Sort by `rrf_score` descending, return top `rerank_top_k` (30).

**Why $k=60$?** The constant dampens the influence of the very top ranks. Reducing $k$ makes rank 1 dominate more; increasing it flattens the distribution. 60 is a well-established empirical value from the original RRF paper.

---

### 4.4 Reranking — Cohere

**File:** `retrieval/reranker.py`

After RRF, the top 30 candidates are **reranked by Cohere's cross-encoder** (`rerank-english-v3.0`):

1. Sends `(query, [chunk_text_1, chunk_text_2, ...])` to the Cohere Rerank API.
2. The cross-encoder scores each (query, document) pair **jointly** — unlike bi-encoders (used for dense retrieval), it can compare query and document simultaneously, yielding much more accurate relevance scores.
3. Returns the top `final_top_k` (5) chunks sorted by `relevance_score`.
4. Adds `rerank_score` field to each chunk dict.

**Why not just use dense retrieval?** Dense embedding models encode query and document independently — they can't model fine-grained word interactions. Cross-encoders are slower but significantly more accurate. The two-stage pipeline (fast cheap retrieval → slow accurate reranking) gets the best of both worlds.

---

## 5. The LLM Layer — Groq with Key Rotation

**File:** `agent/llm.py`

The `get_llm()` function returns a `ChatGroq` (LangChain) instance, always using the **next API key in a round-robin cycle**:

```python
_keys = ["GROQ_KEY_1", "GROQ_KEY_2", "GROQ_KEY_3", "GROQ_KEY_4"]  # up to 4
_key_cycle = itertools.cycle(_keys)
```

Every call to `get_llm()` calls `next(_key_cycle)`, so calls rotate across all configured keys. This is critical because Groq imposes per-key rate limits, and the agent makes **5–7 LLM calls per query** (summarizer, intake, router, query_rewriter, grader, contradiction, answer). With 4 keys, effective throughput is 4× a single key.

**Models used:**
- Primary: `llama-3.3-70b-versatile` (strong reasoning for routing, answering)
- Fallback config: `llama-3.1-8b-instant` (available in config but not auto-switched in current implementation)

**Temperature choices by node:**
| Node | Temperature | Reason |
|---|---|---|
| intake | 0.0 | Deterministic classification |
| safety | 0.0 | Deterministic safety check |
| router | 0.0 | Deterministic routing |
| grader | 0.0 | Deterministic quality check |
| summarizer | 0.2 | Slightly creative summary |
| query_rewriter | 0.1 | Creative but grounded rewriting |
| contradiction | 0.1 | Careful factual analysis |
| answer | 0.2 | Factual, slightly varied phrasing |
| direct_answer | 0.4 | More natural conversational tone |

---

## 6. The Database Layer — Supabase / PostgreSQL

**File:** `api/db.py`

Uses `psycopg2` with a **module-level connection singleton**. The connection is reopened if it's been closed. All queries use `RealDictCursor` — rows are returned as `dict` objects rather than tuples.

`conn.autocommit = True` — no explicit transaction management; each statement commits immediately.

---

### 6.1 Schema

Three tables, created on startup via `ensure_schema()`:

```sql
-- User accounts
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL      PRIMARY KEY,
    username      TEXT        UNIQUE NOT NULL,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- All chat messages (both user and AI turns)
CREATE TABLE IF NOT EXISTS chat_messages (
    id                    BIGSERIAL    PRIMARY KEY,
    session_id            TEXT         NOT NULL,
    user_id               INTEGER      REFERENCES users(id) ON DELETE CASCADE,
    role                  TEXT         NOT NULL,          -- 'human' | 'ai'
    content               TEXT         NOT NULL,
    citations             JSONB        NOT NULL DEFAULT '[]',  -- structured citation objects
    route                 TEXT,                           -- which source was searched
    is_refused            BOOLEAN      DEFAULT FALSE,
    has_contradiction     BOOLEAN      DEFAULT FALSE,
    contradiction_details TEXT,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Composite index for fast session history retrieval
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);

-- Rolling conversation summaries (one row per session)
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id  TEXT        PRIMARY KEY,
    summary     TEXT        NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Key design decision — `citations` as JSONB:** Storing citations as structured JSON inside Postgres means they're queryable but also easily returned as-is to the API without additional JOINs. Each AI message row carries its own citations.

---

### 6.2 All DB Operations

| Function | SQL | Description |
|---|---|---|
| `create_user(username, password)` | `INSERT INTO users ... RETURNING id, username` | Hashes password with bcrypt, inserts. Returns `None` on `UniqueViolation`. |
| `get_user_by_username(username)` | `SELECT id, username, password_hash FROM users WHERE username = %s` | Used during login to fetch the hash for comparison |
| `get_user_by_id(user_id)` | `SELECT id, username FROM users WHERE id = %s` | Used by JWT auth middleware on every protected request |
| `save_message(...)` | `INSERT INTO chat_messages (...)` | Persists one message turn (both human and AI are separate calls) |
| `get_history(session_id)` | `SELECT ... FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC` | Returns all messages for a session in chronological order |
| `list_sessions(user_id)` | Complex `SELECT ... GROUP BY session_id` | Returns one row per session: `{session_id, message_count, preview}`. Preview = first user message (80 chars). |
| `delete_session(session_id)` | `DELETE FROM chat_messages` + `DELETE FROM session_summaries` | Hard deletes all messages and the summary |
| `get_summary(session_id)` | `SELECT summary FROM session_summaries WHERE session_id = %s` | Returns the rolling summary or empty string |
| `save_summary(session_id, text)` | `INSERT ... ON CONFLICT DO UPDATE SET summary = ...` | Upsert — creates or replaces the summary |
| `get_recent_history(session_id, limit)` | Inner `SELECT ... ORDER BY created_at DESC LIMIT N` → outer `ORDER BY created_at ASC` | Returns the last N messages in chronological order (the inner DESC + outer ASC pattern gets the "last N" correctly ordered) |
| `count_messages(session_id)` | `SELECT COUNT(*) FROM chat_messages WHERE session_id = %s` | Used by summarizer_node to decide whether to summarize |

**Password hashing:** Uses `bcrypt` directly (not through `passlib`) to avoid a version-compatibility warning (`bcrypt >= 4` changed internal APIs that `passlib` hadn't caught up with). `bcrypt.hashpw(plain.encode(), bcrypt.gensalt())` with bcrypt's built-in salt generation.

---

### 6.3 Chat History Strategy (Summary + Recent)

The `run_agent()` function in `graph.py` builds `history_context` **before** invoking the graph:

```python
chat_summary, history_context = _build_history_context(session_id)
```

**Strategy:**
```
if chat_summary exists (history was long enough to summarize):
    history_context = "[Summary of earlier conversation]\n{summary}\n\n[Recent messages]\n{last 4 messages}"
else:
    history_context = "[Recent messages]\n{last 6 messages}"
```

This adaptive approach:
- **Short sessions (< 10 msgs):** Full recent history, no summary overhead.
- **Long sessions (≥ 10 msgs):** Compressed summary of older turns + the last few raw turns. The LLM gets enough context to resolve references without consuming excessive tokens.

**Why not just use LangGraph's built-in message history?** LangGraph's checkpointer stores messages, but they would all be replayed into every prompt without compression. For medical conversations that can run 20–50 turns, this would quickly exhaust token limits. The custom Postgres + summary strategy keeps context window usage bounded.

---

### 6.4 LangGraph Checkpointer

**File:** `agent/graph.py` → `get_postgres_checkpointer()`

Uses `langgraph-checkpoint-postgres` → `PostgresSaver.from_conn_string(SUPABASE_DB_URL)`.

This creates LangGraph's own checkpoint tables in Postgres (separate from the `chat_messages` table). The checkpointer stores the full `AgentState` JSON after every graph invocation, keyed by `thread_id = session_id`.

This is used for LangGraph's **internal state continuity** (e.g., keeping the `messages` list across turns). The `chat_messages` table is used for **user-facing history** (displaying conversations in the UI, building `history_context`). Both mechanisms coexist.

---

## 7. The API Layer — FastAPI

**File:** `api/main.py`

### 7.1 Authentication Flow

MedRAG uses **JWT (JSON Web Token)** authentication via `PyJWT`:

**Registration (`POST /api/auth/register`):**
1. Validates `RegisterRequest` (username 3-40 chars, password min 6 chars) via Pydantic.
2. Calls `db.create_user(username, password)` — returns `None` if username taken.
3. If taken → HTTP 409 Conflict.
4. Otherwise → creates a JWT with payload `{"sub": user_id, "username": username, "exp": now + 30 days}`.
5. Returns `{access_token, token_type: "bearer", username}`.

**Login (`POST /api/auth/login`):**
1. Fetches user by username (includes `password_hash`).
2. If not found or `bcrypt.checkpw(plain, hash)` fails → HTTP 401.
3. Creates and returns JWT.

**Protected routes** use a FastAPI dependency:
```python
async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)):
    payload = _decode_token(creds.credentials)  # validates signature + expiry
    user = db.get_user_by_id(int(payload["sub"]))
    if not user: raise HTTPException(401)
    return user
```

The `_bearer = HTTPBearer(auto_error=False)` extracts the `Authorization: Bearer <token>` header. `auto_error=False` lets the dependency return `None` for missing headers, allowing a custom error message.

---

### 7.2 Chat Endpoint Flow

**`POST /api/chat`** (protected):

```
1. Validate ChatRequest (session_id, message, new_session)
2. Check message is not empty
3. Get pre-initialized LangGraph agent (lazy singleton)
4. Call run_agent(query, session_id, agent)
   → builds history_context from Postgres
   → invokes the full LangGraph graph
   → returns final AgentState dict
5. Build CitationModel list from result["citations"]
6. Save HumanMessage to chat_messages table
7. Save AIMessage with citations, route, is_refused, has_contradiction to chat_messages table
8. Return ChatResponse with answer, citations, route, flags, stats
```

**Important:** Messages are saved **after** the agent returns. This means if the agent crashes mid-run, no partial message is saved to the DB.

**`ChatResponse` fields returned:**
- `session_id` — echo back
- `answer` — the full answer text (may be a refusal message)
- `citations` — list of structured `CitationModel` objects
- `route` — which source was searched (`research/guideline/both`)
- `is_refused` — whether safety node refused the query
- `refuse_reason` — reason string if refused
- `has_contradiction` — whether sources conflicted
- `contradiction_details` — description of the conflict
- `rewrite_count` — how many query rewrites occurred (0, 1, or 2)
- `docs_retrieved` — number of docs retrieved (0 if refused/conversational)

---

### 7.3 Session Management Endpoints

| Endpoint | Implementation |
|---|---|
| `GET /api/sessions` | Calls `db.list_sessions(user_id)` — returns `[{session_id, message_count, preview}]` sorted by most recent activity |
| `GET /api/sessions/{id}/history` | Calls `db.get_history(session_id)` — returns all messages with citations, ordered chronologically |
| `DELETE /api/sessions/{id}` | Calls `db.delete_session(session_id)` — hard deletes messages + summary |

Sessions are **implicitly created** when a `session_id` is used for the first time in `/api/chat`. There is no "create session" endpoint — the client generates a UUID and starts using it.

### 7.4 Pydantic Models

**File:** `api/models.py`

All request/response shapes are enforced with Pydantic v2. Key models:

| Model | Fields | Notes |
|---|---|---|
| `ChatRequest` | `session_id`, `message`, `new_session` | `new_session` not currently used for special logic |
| `CitationModel` | `index`, `id`, `title`, `source_type`, `condition`, `url`, `pub_date`, `pmid`, `snippet`, `rerank_score` | `snippet` is 200 chars of chunk text |
| `ChatResponse` | All chat result fields | Includes contradiction + safety flags |
| `RegisterRequest` | `username` (3-40 chars), `password` (min 6 chars) | Pydantic field validators |
| `TokenResponse` | `access_token`, `token_type`, `username` | Returned on register/login |
| `HealthResponse` | `status`, `qdrant`, `groq_keys`, `bm25_index` | System component status strings |

**CORS:** The `ALLOWED_ORIGINS` environment variable accepts comma-separated origins. Defaults include the GitHub Pages URL and localhost. The middleware allows credentials (needed for Authorization headers) and all HTTP methods.

---

## 8. Frontend — How the UI Works

**Files:** `frontend/index.html`, `frontend/app.js`, `frontend/style.css`

The frontend is a **single-page application** with no build step, framework, or bundler.

**Key behaviors in `app.js`:**

1. **Auth state management:** JWT token stored in `localStorage`. On load, calls `/api/auth/verify` with the stored token. If valid → shows chat UI. If invalid/missing → shows login modal.

2. **Session management:** On first load after auth, calls `GET /api/sessions` to populate the sidebar. Sessions are displayed with their preview (first user message) and message count.

3. **New session creation:** The client generates a UUID (`crypto.randomUUID()`) as the `session_id`. When the user clicks "New Chat", a new UUID is generated and used for subsequent messages.

4. **Chat submission:** On send, POSTs to `/api/chat` with `{session_id, message}`. While waiting, shows a loading indicator. On response:
   - Renders the answer text in a message bubble
   - If `has_contradiction=True`, shows a warning banner about conflicting sources
   - Renders citations as expandable cards showing title, source type, snippet, and link

5. **Thinking indicator:** Displays an animated "thinking" panel while waiting for the backend, giving the user visual feedback during the multiple LLM calls (typically 3–7 seconds total).

6. **History loading:** Clicking a session in the sidebar calls `GET /api/sessions/{id}/history` and re-renders all messages with their citations.

---

## 9. Configuration System

**File:** `config.py`

All tunable parameters live in one place as Python dataclasses. The singleton `cfg = Config()` is imported throughout the codebase. No scattered magic numbers.

| Config Class | Key Parameters |
|---|---|
| `LLMConfig` | `model`, `temperature`, `max_tokens`, `max_retries` |
| `EmbeddingConfig` | `model` (bge-base-en-v1.5), `dimension` (768), `batch_size` (64) |
| `QdrantConfig` | `collection_name`, `distance_metric`, `dense_top_k` (30) |
| `BM25Config` | `top_k` (30), `k1` (1.5), `b` (0.75), `index_path` |
| `RetrievalConfig` | `rrf_k` (60), `rerank_top_k` (30), `final_top_k` (5), `cohere_rerank_model` |
| `ChunkConfig` | `chunk_size` (400 tokens), `chunk_overlap` (50 tokens) |
| `IngestionConfig` | `conditions`, `pubmed_queries`, `pubmed_max_per_condition` (350) |
| `AgentConfig` | `max_rewrite_retries` (2), `relevance_threshold` (0.5), `chat_history_window` (10) |

All values can be overridden at runtime via environment variables where applicable (e.g. `QDRANT_COLLECTION_NAME` overrides `cfg.qdrant.collection_name`).

---

## 10. End-to-End Request Walkthrough

**User asks:** *"What are the treatment options for type 2 diabetes according to recent research?"*

**Assume:** Authenticated user, existing session with 3 prior messages.

```
1. POST /api/chat
   └── FastAPI validates JWT → user_id=42
   └── run_agent(query, session_id)

2. _build_history_context(session_id)
   └── count_messages() → 6 (below threshold of 10)
   └── get_recent_history(limit=6) → 3 Q+A pairs
   └── history_context = "[Recent messages]\nUser: ...\nAssistant: ..."

3. LangGraph.invoke(initial_state, config={"thread_id": session_id})

4. summarizer_node
   └── count_messages() = 6 < 10 → no-op
   └── returns {chat_summary: ""}

5. intake_node
   └── LLM call: "type 2 diabetes treatment options" → RAG
   └── returns {needs_rag: True}

6. safety_node
   └── regex patterns: no match ("what are the treatment options" is educational)
   └── LLM call → SAFE
   └── returns {is_safe: True}

7. router_node
   └── LLM call: "research" keyword matches "recent research"
   └── returns {route: "research"}

8. query_rewriter_node
   └── LLM call: resolves context, single topic
   └── Output: "QUERIES:\n1. type 2 diabetes treatment management clinical evidence"
   └── returns {search_queries: ["type 2 diabetes treatment management clinical evidence"],
                search_query: "type 2 diabetes treatment management clinical evidence"}

9. (single query → retriever_node)

10. retriever_node
    └── dense_retriever.search("type 2 diabetes treatment...", source_type="research", top_k=30)
        └── embed_one(query) → 768-dim vector via Cloudflare API
        └── qdrant.query_points(vector, filter=source_type=="research", limit=30) → 30 docs
    └── bm25_retriever.search(query, source_type="research", top_k=30) → 30 docs
    └── hybrid_retriever: RRF fusion → 30 merged candidates (many appear in both lists)
    └── reranker.rerank(query, 30 candidates, top_k=5) → top 5 chunks
    └── returns {retrieved_docs: [chunk1, chunk2, chunk3, chunk4, chunk5]}

11. grader_node
    └── LLM call with top-5 chunks: "Do these answer the question?" → yes
    └── returns {docs_relevant: True, rewrite_count: 0}

12. contradiction_node
    └── source_type="research" only → source1=[RESEARCH], source2=[RESEARCH], ...
    └── LLM call → "NO_CONFLICT" (all from same source type, same direction)
    └── returns {has_contradiction: False}

13. answer_node
    └── formats 5 sources as [1] (RESEARCH) Title\nText...
    └── LLM call (temperature=0.2, max_tokens=1024):
        "Answer the question using ONLY the provided sources..."
    └── LLM returns: "Recent research highlights several treatment approaches for
        type 2 diabetes [1][3]. Metformin remains the first-line agent [2]..."
    └── builds citations list with index, title, url, pmid, snippet, rerank_score
    └── returns {answer: "...", citations: [...], messages: [HumanMsg, AIMsg]}

14. run_agent() returns final_state

15. FastAPI saves:
    └── db.save_message(session_id, user_id, role="human", content=query)
    └── db.save_message(session_id, user_id, role="ai", content=answer,
                        citations=citations, route="research", ...)

16. Returns ChatResponse to frontend:
    {
      session_id: "...",
      answer: "Recent research highlights...",
      citations: [{index:1, title:"...", url:"https://pubmed.ncbi.nlm.nih.gov/...", ...}, ...],
      route: "research",
      is_refused: false,
      has_contradiction: false,
      rewrite_count: 0,
      docs_retrieved: 5
    }

17. Frontend renders answer in chat bubble + citation cards
```

**Total LLM calls for this query: 5**  
(intake, safety, router, query_rewriter, grader, answer — contradiction always runs but is one more)  
**Total time:** ~3–6 seconds depending on Groq API latency.
