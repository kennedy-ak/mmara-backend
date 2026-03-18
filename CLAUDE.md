1# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMara Backend is a multi-agent RAG (Retrieval-Augmented Generation) system for Ghanaian legal assistance. It uses FastAPI with async SQLAlchemy (PostgreSQL), Redis for caching/sessions, Pinecone for vector search, and OpenAI for embeddings and LLM responses.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or: uv pip install -r requirements.txt

# Run dev server (auto-reload)
uvicorn app.main:app --reload

# Run with Docker (full stack)
docker-compose up -d
docker-compose exec app python scripts/seed_db.py
docker-compose exec app python scripts/process_docs.py

# Local dev with dockerized services
docker-compose up -d postgres redis
python scripts/seed_db.py
python scripts/process_docs.py
uvicorn app.main:app --reload

# Tests
pytest
pytest --cov=app --cov-report=html
pytest tests/test_agents.py -v       # single test file

# Document processing
python scripts/process_docs.py [--rebuild] [-d /path] [-c category]
```

## Architecture

### Multi-Agent Pipeline

User queries flow through a chain of specialized agents orchestrated by `AgentOrchestrator` (`app/agents/orchestrator.py`):

```
Query → IntakeAgent → SafetyAgent → LegalAgent → [EmergencyHandler] → LLM Response → ResponseValidator
```

1. **IntakeAgent** (`app/agents/intake.py`) — Classifies intent (question/emergency/clarification/general), category (criminal/road_traffic/general), and urgency level
2. **SafetyAgent** (`app/agents/safety.py`) — Filters restricted content, detects emergencies, enforces disclaimers
3. **LegalAgent** (`app/agents/legal.py`) — Formulates retrieval queries, performs hybrid search (semantic + BM25), extracts citations
4. **EmergencyHandler** (`app/agents/safety.py`) — Activated for high-urgency queries with immediate guidance
5. **ResponseValidator** (`app/agents/safety.py`) — Validates citation presence and disclaimer inclusion

### Hybrid Retrieval

`RetrievalService` (`app/services/retrieval.py`) combines:
- **Semantic search**: Pinecone vectors via OpenAI `text-embedding-3-small` (1536-dim)
- **BM25 keyword search**: rank-bm25 library
- **Reciprocal Rank Fusion (RRF)** to merge results, with optional OpenAI reranking

Configurable via `RETRIEVAL_ALPHA` (semantic weight), `TOP_K_RETRIEVAL`, and `RRF_K`.

### API Structure

Base URL: `/api/v1`. Routers in `app/api/v1/`:
- **auth.py** — Register, login (OAuth2 form + JSON), JWT refresh, password reset
- **chat.py** — Query submission, session history, feedback, WebSocket streaming
- **admin.py** — Document upload/management, reindexing, stats (admin role required)
- **users.py** — Profile management, user analytics

Health/info endpoints at root: `GET /`, `/health`, `/metrics`, `/info`.

### Authentication

JWT Bearer tokens. 24-hour access tokens, 7-day refresh tokens. Role-based access: `admin`, `user`, `free`. Rate limiting per role via Redis.

### Database

Async SQLAlchemy 2.0 with asyncpg. Tables auto-created on startup via `init_db()` (no Alembic). Key models in `app/db/models.py`: `User`, `ChatSession`, `Document`, `Analytics`, `RateLimit`, `PasswordReset`.

### Key Services

| Service | File | Purpose |
|---------|------|---------|
| OpenAIClient | `app/services/openai_client.py` | Chat completions, streaming, embeddings, classification |
| EmbeddingService | `app/services/embeddings.py` | Pinecone vector DB management, embedding generation |
| RetrievalService | `app/services/retrieval.py` | Hybrid search with RRF fusion |
| LegalDocumentChunker | `app/services/chunker.py` | PDF parsing → chunks with metadata |
| RedisService | `app/services/redis_client.py` | Session history, caching |
| EmailService | `app/services/email_service.py` | Password reset emails via SMTP |

### Configuration

All settings via environment variables, validated by Pydantic Settings in `app/config.py`. See `.env.example` for the full list. Key groups: database (`DATABASE_URL`), Redis (`REDIS_URL`), OpenAI (`OPENAI_API_KEY`, `OPENAI_MODEL`), Pinecone (`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`), retrieval tuning parameters, and rate limits.

Legal system prompts and emergency templates are defined in `app/config.py`.

### Streaming

`StreamingOrchestrator` in `app/agents/orchestrator.py` supports WebSocket streaming at `WS /api/v1/chat/stream`.

## Tech Stack

Python 3.12+, FastAPI, SQLAlchemy 2.0 (async/asyncpg), PostgreSQL 16, Redis 7, Pinecone, OpenAI API (gpt-4o-mini default), LlamaParse for PDF extraction, LangSmith for tracing, pytest for testing.
