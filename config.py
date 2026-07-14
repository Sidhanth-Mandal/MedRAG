# ============================================================
#  config.py  —  Central configuration for the Agentic RAG system
#  Edit this file to tune model names, retrieval params, etc.
#  All values here can be overridden by .env if needed.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field


# ── LLM / Groq ──────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    # Primary chat model used for routing, grading, answer generation
    model: str = "llama-3.3-70b-versatile"

    # Fallback model if primary hits rate limits (optional — same rotation keys)
    fallback_model: str = "llama-3.1-8b-instant"

    # Generation parameters
    temperature: float = 0.2          # Low for factual medical Q&A
    max_tokens: int = 1024            # Answer length cap
    top_p: float = 0.9

    # Retry / back-off settings for Groq API calls
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


# ── Embeddings / Cloudflare Workers AI ─────────────────────────────────────

@dataclass
class EmbeddingConfig:
    model: str = "@cf/baai/bge-base-en-v1.5"
    dimension: int = 768              # Output vector size for bge-base-en-v1.5
    batch_size: int = 64              # Max texts per Cloudflare API call


# ── Qdrant Vector Store ─────────────────────────────────────────────────────

@dataclass
class QdrantConfig:
    collection_name: str = "medical_rag"
    distance_metric: str = "Cosine"   # "Cosine" | "Dot" | "Euclid"
    # How many dense candidates to pull before reranking
    dense_top_k: int = 30


# ── BM25 / Sparse Retrieval ─────────────────────────────────────────────────

@dataclass
class BM25Config:
    # How many BM25 candidates to retrieve before fusion
    top_k: int = 30

    # BM25 hyperparameters
    k1: float = 1.5
    b: float = 0.75

    # Path where the serialized BM25 index is saved (relative to project root)
    index_path: str = "data/bm25_index.pkl"


# ── Hybrid Retrieval & Reranking ────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    # Reciprocal Rank Fusion constant
    rrf_k: int = 60

    # Number of candidates passed to Cohere reranker after RRF fusion
    rerank_top_k: int = 30

    # Final number of chunks kept after reranking (sent to LLM)
    final_top_k: int = 5

    # Cohere rerank model
    cohere_rerank_model: str = "rerank-english-v3.0"


# ── Chunking ────────────────────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    chunk_size: int = 400             # Target tokens per chunk
    chunk_overlap: int = 50           # Overlap between consecutive chunks
    # Tokenizer used for size estimation ("cl100k_base" works for most models)
    tokenizer: str = "cl100k_base"


# ── Data Ingestion ───────────────────────────────────────────────────────────

@dataclass
class IngestionConfig:
    # Medical conditions to pull data for
    conditions: list[str] = field(default_factory=lambda: [
        "type2_diabetes",
        "hypertension",
        "asthma",
    ])

    # PubMed search terms mapped to each condition slug
    pubmed_queries: dict[str, str] = field(default_factory=lambda: {
        "type2_diabetes": "type 2 diabetes mellitus management treatment",
        "hypertension":   "hypertension treatment blood pressure management",
        "asthma":         "asthma treatment management inhaler",
    })

    # MedlinePlus Health Topics search terms per condition
    medlineplus_queries: dict[str, str] = field(default_factory=lambda: {
        "type2_diabetes": "diabetes type 2",
        "hypertension":   "high blood pressure hypertension",
        "asthma":         "asthma",
    })

    # Max PubMed abstracts to fetch per condition
    pubmed_max_per_condition: int = 350   # ~350 × 3 = ~1050 total

    # PubMed date range (YYYY/MM/DD)
    pubmed_date_start: str = "2019/01/01"
    pubmed_date_end:   str = "2025/12/31"

    # Output paths (relative to project root)
    pubmed_output_path:     str = "data/raw/pubmed_abstracts.json"
    medlineplus_output_path: str = "data/raw/medlineplus_guidelines.json"


# ── Agent / LangGraph ────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    # Maximum times the grader can trigger a query rewrite before giving up
    max_rewrite_retries: int = 2

    # Minimum relevance score (0-1) from grader to accept retrieved docs
    relevance_threshold: float = 0.5

    # How many recent messages to keep in the raw chat buffer before summarizing
    chat_history_window: int = 10


# ── Master Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    llm:        LLMConfig       = field(default_factory=LLMConfig)
    embedding:  EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant:     QdrantConfig    = field(default_factory=QdrantConfig)
    bm25:       BM25Config      = field(default_factory=BM25Config)
    retrieval:  RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk:      ChunkConfig     = field(default_factory=ChunkConfig)
    ingestion:  IngestionConfig = field(default_factory=IngestionConfig)
    agent:      AgentConfig     = field(default_factory=AgentConfig)


# ── Singleton instance (import this everywhere) ─────────────────────────────
cfg = Config()
