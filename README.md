# 🩺 MedRAG — Agentic Medical Q&A System

An **Agentic Retrieval-Augmented Generation (RAG)** chatbot for medical Q&A, powered by a LangGraph multi-step reasoning agent, a hybrid retrieval pipeline (dense + BM25 + reranking), and a clean vanilla-JS frontend. The backend is deployed on **Google Cloud Run**; the frontend is hosted on **GitHub Pages**.

---

## 📌 Table of Contents

- [Project Purpose](#project-purpose)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Backend Setup](#backend-setup)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Local Development](#local-development)
  - [Data Ingestion & Pipeline](#data-ingestion--pipeline)
  - [Docker & Cloud Run](#docker--cloud-run)
- [Frontend Setup](#frontend-setup)
  - [Local Development](#local-development-1)
  - [Pointing to the Backend](#pointing-to-the-backend)
  - [GitHub Pages Deployment](#github-pages-deployment)
- [API Reference](#api-reference)
- [CI/CD](#cicd)

---

## Project Purpose

MedRAG answers medical questions about **Type 2 Diabetes**, **Hypertension**, and **Asthma** by grounding responses in real evidence:

- **PubMed abstracts** (research literature, ~350 per condition)
- **MedlinePlus guidelines** (patient-facing clinical guidelines)

Rather than a naive single-step RAG, it uses a **LangGraph state machine** that:

1. Summarises conversation history to stay context-aware across sessions
2. Classifies the query (conversational vs. medical)
3. Runs a safety check to filter harmful requests
4. Routes to the most relevant knowledge source (research / guideline / both)
5. Rewrites and expands queries if retrieval quality is low
6. Grades retrieved documents for relevance, retrying if needed
7. Detects contradictions between retrieved sources and flags them
8. Generates a cited, factual answer

All chat sessions are persisted in **Supabase (PostgreSQL)** with JWT-based user authentication.

---

## Architecture Overview

```
Frontend (GitHub Pages)
        │  HTTPS / REST
        ▼
FastAPI Backend (Google Cloud Run)
        │
        ├── Auth layer (JWT + Supabase users table)
        │
        └── LangGraph Agent
                │
                ├── summarizer → intake → safety → router
                │
                ├── query_rewriter → [single | multi] retriever
                │         ↑                    │
                │         └── grade (retry) ◄──┘
                │
                └── contradiction → answer
                        │
                        ├── Qdrant (dense vector search, Cloudflare embeddings)
                        └── BM25  (sparse keyword search)
                                │
                        Cohere Reranker (fusion + rerank)
                                │
                        Groq LLaMA 3.3 70B (generation)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — LLaMA 3.3 70B Versatile (with round-robin key rotation) |
| **Agent** | LangGraph state machine |
| **Embeddings** | Cloudflare Workers AI — `bge-base-en-v1.5` (768-dim) |
| **Vector Store** | Qdrant Cloud |
| **Sparse Retrieval** | BM25 (`rank-bm25`) |
| **Reranker** | Cohere Rerank v3 |
| **API** | FastAPI + Uvicorn |
| **Database** | Supabase (PostgreSQL) — chat history + users |
| **Checkpointing** | LangGraph PostgresSaver |
| **Data Sources** | PubMed (Entrez/Biopython), MedlinePlus |
| **Auth** | JWT (PyJWT) + bcrypt (passlib) |
| **Frontend** | Vanilla HTML/CSS/JavaScript |
| **Backend Deploy** | Google Cloud Run + Artifact Registry |
| **Frontend Deploy** | GitHub Pages |
| **CI/CD** | GitHub Actions |

---

## Folder Structure

```
Agentic RAG/
│
├── .github/
│   └── workflows/
│       ├── deploy-backend.yml      # CI/CD: build Docker → push to Artifact Registry → deploy to Cloud Run
│       └── deploy-frontend.yml     # CI/CD: push frontend/ folder to GitHub Pages
│
├── agent/                          # LangGraph agent (the brain)
│   ├── graph.py                    # Builds & compiles the state machine, run_agent() entrypoint
│   ├── llm.py                      # Groq LLM client with round-robin API key rotation
│   ├── state.py                    # AgentState TypedDict (shared across all graph nodes)
│   └── nodes/
│       ├── summarizer.py           # Condenses chat history into a rolling summary
│       ├── intake.py               # Classifies query: conversational vs. RAG-needed
│       ├── direct_answer.py        # Fast-path: answers conversational turns without RAG
│       ├── safety.py               # Filters unsafe / harmful medical requests
│       ├── router.py               # Decides which knowledge source to search (research/guideline/both)
│       ├── query_rewriter.py       # Expands/rewrites the query for better retrieval
│       ├── retriever.py            # Single-query hybrid retrieval (dense + BM25 + rerank)
│       ├── multi_retriever.py      # Multi-query retrieval (runs retriever per sub-query)
│       ├── grader.py               # Scores retrieved docs for relevance; triggers retry loop
│       ├── contradiction.py        # Detects conflicts between research vs. guideline sources
│       └── answer.py               # Synthesises final answer with inline citations
│
├── api/                            # FastAPI REST layer
│   ├── main.py                     # All route definitions (auth, chat, sessions, health)
│   ├── models.py                   # Pydantic request/response models
│   └── db.py                       # Supabase/Postgres CRUD (users, messages, sessions, summaries)
│
├── ingestion/                      # Raw data fetchers
│   ├── pubmed_fetcher.py           # Downloads PubMed abstracts via Entrez API
│   ├── medlineplus_fetcher.py      # Scrapes MedlinePlus health topic pages
│   └── run_ingestion.py            # CLI entry point: run both fetchers
│
├── pipeline/                       # Data processing & vector store upload
│   ├── chunker.py                  # Token-aware text chunking (tiktoken)
│   ├── embedder.py                 # Cloudflare Workers AI embedding calls
│   ├── qdrant_uploader.py          # Uploads embedded chunks to Qdrant Cloud
│   ├── run_pipeline.py             # CLI entry point: chunk → embed → upload
│   └── verify_search.py            # Quick sanity-check query against Qdrant
│
├── retrieval/                      # Retrieval strategies
│   ├── dense_retriever.py          # Qdrant dense vector search
│   ├── bm25_retriever.py           # BM25 index build & search
│   ├── hybrid_retriever.py         # RRF fusion of dense + BM25, then Cohere rerank
│   ├── reranker.py                 # Cohere Rerank API wrapper
│   └── compare_retrieval.py        # Dev utility: compare retrieval strategies
│
├── eval/                           # Offline evaluation
│   ├── test_cases.py               # Hand-crafted Q&A test cases per condition
│   └── run_eval.py                 # Runs the agent over test cases, prints scores
│
├── frontend/                       # Static web UI (deployed to GitHub Pages)
│   ├── index.html                  # App shell, login/register modals, chat layout
│   ├── app.js                      # All client-side logic (auth, chat, sessions, UI)
│   └── style.css                   # Full design system, dark mode, animations
│
├── data/                           # Local data artefacts (git-ignored)
│   ├── raw/                        # JSON dumps from ingestion
│   │   ├── pubmed_abstracts.json
│   │   └── medlineplus_guidelines.json
│   └── bm25_index.pkl              # Serialised BM25 index (built by pipeline)
│
├── config.py                       # Central configuration (models, retrieval params, etc.)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Production container (python:3.11-slim)
├── .env.example                    # Template for all required environment variables
├── .env                            # Your actual secrets (never commit this!)
└── .gitignore
```

---

## Backend Setup

### Prerequisites

- Python 3.11+
- A virtual environment tool (`venv` or `conda`)
- Accounts / API keys for: **Qdrant Cloud**, **Cloudflare Workers AI**, **Cohere**, **Groq** (×4 keys for rotation), **Supabase**, **NCBI/Entrez**

### Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `ENTREZ_EMAIL` | Your email — required by NCBI to use the Entrez API |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Collection name (default: `medical_rag`) |
| `CF_ACCOUNT_ID` | Cloudflare account ID (for embeddings) |
| `CF_API_TOKEN` | Cloudflare API token |
| `COHERE_API_KEY` | Cohere API key (reranking) |
| `GROQ_API_KEY_1` … `_4` | Groq API keys — up to 4 for round-robin rotation |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase anon or service role key |
| `SUPABASE_DB_URL` | PostgreSQL connection string for LangGraph checkpointer |
| `JWT_SECRET` | A long random secret string for signing JWTs |
| `JWT_ALGORITHM` | Default: `HS256` |
| `JWT_EXPIRE_DAYS` | Token TTL in days (default: `30`) |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins |

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd "Agentic RAG"

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your actual API keys

# 5. Start the API server
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Data Ingestion & Pipeline

> **Run these once** to populate the Qdrant vector store before the chatbot can answer questions.

```bash
# Step 1 — Fetch raw data from PubMed and MedlinePlus
python -m ingestion.run_ingestion

# Step 2 — Chunk, embed, and upload to Qdrant
python -m pipeline.run_pipeline

# (Optional) Verify the vector store is populated
python -m pipeline.verify_search
```

This will download ~1,050 PubMed abstracts (350 per condition) and MedlinePlus guidelines for Type 2 Diabetes, Hypertension, and Asthma, then upload them as embedded vectors to Qdrant and build a local BM25 index at `data/bm25_index.pkl`.

### Docker & Cloud Run

**Build and run locally with Docker:**

```bash
docker build -t medrag-api .

docker run -p 8080:8080 \
  --env-file .env \
  medrag-api
```

**Deploy to Google Cloud Run (manually):**

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push image
gcloud builds submit --tag asia-south1-docker.pkg.dev/YOUR_PROJECT/med-rag/medrag-api

# Deploy
gcloud run deploy medrag-api \
  --image asia-south1-docker.pkg.dev/YOUR_PROJECT/med-rag/medrag-api \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars "QDRANT_URL=...,GROQ_API_KEY_1=...,..."
```

> ⚠️ In production, set all `.env` variables as Cloud Run **Secret Manager** secrets or environment variables rather than passing them on the command line.

---

## Frontend Setup

The frontend is a **pure static site** (HTML + CSS + JS — no build step required).

### Local Development

Simply open `frontend/index.html` directly in a browser, **or** serve it with any static server to avoid CORS issues:

```bash
# Using Python's built-in server (from the project root)
python -m http.server 5500 --directory frontend

# Or with Node.js
npx serve frontend
```

Then open `http://localhost:5500`.

### Pointing to the Backend

The frontend reads the backend URL from a `API_BASE` constant defined near the top of [`frontend/app.js`](frontend/app.js).

- **Local development:** Set `API_BASE` to `http://localhost:8000`
- **Production:** Set `API_BASE` to your Cloud Run service URL

```js
// frontend/app.js  (near the top)
const API_BASE = "https://your-cloud-run-url.a.run.app";
```

Make sure your Cloud Run service has the frontend's GitHub Pages URL in its `ALLOWED_ORIGINS` environment variable to avoid CORS errors.

### GitHub Pages Deployment

The frontend is automatically deployed to GitHub Pages on every push to `main` that touches the `frontend/` folder.

To deploy manually:

1. Go to your repository → **Settings → Pages**
2. Set source to **GitHub Actions**
3. Trigger the `Deploy Frontend to GitHub Pages` workflow from the **Actions** tab

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/register` | — | Register a new user, returns JWT |
| `POST` | `/api/auth/login` | — | Login, returns JWT |
| `GET` | `/api/auth/verify` | ✅ Bearer | Verify token, return current user |
| `POST` | `/api/chat` | ✅ Bearer | Send a message, get AI answer + citations |
| `GET` | `/api/sessions` | ✅ Bearer | List the authenticated user's sessions |
| `GET` | `/api/sessions/{id}/history` | ✅ Bearer | Full chat history for a session |
| `DELETE` | `/api/sessions/{id}` | ✅ Bearer | Delete a session and its history |
| `GET` | `/api/health` | — | Health check (Qdrant, BM25, Groq key count) |
| `GET` | `/api/stats` | — | Document counts from Qdrant by source/condition |

Full interactive docs are available at `<backend-url>/docs` (Swagger UI).

---

## CI/CD

Two GitHub Actions workflows automate deployments on every push to `main`:

### Backend — `.github/workflows/deploy-backend.yml`

Triggers on pushes to `main` that **do not** exclusively touch `frontend/` or `.md` files.

1. Authenticates to Google Cloud via **Workload Identity Federation** (no long-lived credentials stored)
2. Builds a Docker image tagged with the Git SHA
3. Pushes to **Google Artifact Registry**
4. Deploys to **Google Cloud Run** in `asia-south1`

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `WIF_PROVIDER` | Workload Identity Federation provider resource name |
| `WIF_SERVICE_ACCOUNT` | GCP service account email for deployment |

### Frontend — `.github/workflows/deploy-frontend.yml`

Triggers on pushes to `main` that **only** touch the `frontend/` folder.

1. Uploads the `frontend/` directory as a GitHub Pages artifact
2. Deploys to GitHub Pages

No additional secrets required beyond the default `GITHUB_TOKEN`.

---

## Notes

- The **BM25 index** (`data/bm25_index.pkl`) is built locally and baked into the Docker image at build time. Re-run the pipeline and redeploy if you add new documents.
- The **LangGraph checkpointer** uses Supabase's PostgreSQL connection string. If `SUPABASE_DB_URL` is not set, the agent runs without persistence (no cross-session memory).
- Groq enforces rate limits — `llm.py` rotates across up to **4 API keys** automatically to maximise throughput.
