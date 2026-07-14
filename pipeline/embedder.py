"""
pipeline/embedder.py
─────────────────────
Cloudflare Workers AI embedding client.

Model: @cf/baai/bge-base-en-v1.5  (768-dim)
Docs:  https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/

Batches text inputs, calls the REST API, returns float vectors.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

load_dotenv()

_CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
_CF_API_TOKEN  = os.environ.get("CF_API_TOKEN",  "")

_CF_EMBED_URL  = (
    "https://api.cloudflare.com/client/v4/accounts/"
    "{account_id}/ai/run/{model}"
)


class CloudflareEmbedder:
    """Wraps the Cloudflare Workers AI REST API for batch embedding."""

    def __init__(
        self,
        account_id: str = _CF_ACCOUNT_ID,
        api_token:  str = _CF_API_TOKEN,
        model:      str = cfg.embedding.model,
        batch_size: int = cfg.embedding.batch_size,
    ):
        if not account_id:
            raise ValueError("CF_ACCOUNT_ID not set in environment")
        if not api_token:
            raise ValueError("CF_API_TOKEN not set in environment")

        self.account_id = account_id
        self.api_token  = api_token
        self.model      = model
        self.batch_size = batch_size
        self.dimension  = cfg.embedding.dimension

        self._url = _CF_EMBED_URL.format(
            account_id=account_id,
            model=model,
        )
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type":  "application/json",
        }

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns a list of float vectors.
        Automatically retries on transient errors (429 / 5xx).
        """
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._call_api(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper."""
        return self.embed_batch([text])[0]

    def _call_api(
        self,
        texts: list[str],
        max_retries: int = 3,
        backoff: float = 2.0,
    ) -> list[list[float]]:
        """POST to CF Workers AI; retry on 429 / 5xx."""
        payload = {"text": texts}

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        self._url,
                        headers=self._headers,
                        json=payload,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    # CF returns: {"result": {"data": [[...], [...]]}, "success": true}
                    vectors = data["result"]["data"]
                    return vectors

                elif resp.status_code == 429:
                    wait = backoff * (2 ** attempt)
                    print(f"  [RATE_LIMIT] CF rate-limit hit, waiting {wait:.1f}s …")
                    time.sleep(wait)

                else:
                    print(f"  [WARN] CF API returned {resp.status_code}: {resp.text[:200]}")
                    time.sleep(backoff)

            except httpx.RequestError as exc:
                print(f"  [WARN] Request error (attempt {attempt+1}): {exc}")
                time.sleep(backoff)

        raise RuntimeError(
            f"Cloudflare embedding API failed after {max_retries} retries"
        )


# ── Singleton for convenience ─────────────────────────────────────────────────
_embedder: CloudflareEmbedder | None = None


def get_embedder() -> CloudflareEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CloudflareEmbedder()
    return _embedder


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embedder = get_embedder()
    test_texts = [
        "What is the treatment for type 2 diabetes?",
        "High blood pressure management guidelines.",
        "Asthma inhaler usage in adults.",
    ]
    print("Embedding test texts …")
    vectors = embedder.embed_batch(test_texts)
    print(f"[OK] Got {len(vectors)} vectors, dim={len(vectors[0])}")
    print(f"     First vector (first 5 dims): {vectors[0][:5]}")
