# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps (psycopg2-binary needs libpq) ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────
COPY . .

# ── Cloud Run sets PORT env var (default 8080) ─────────────────
ENV PORT=8080
EXPOSE 8080

# ── Start server ───────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]