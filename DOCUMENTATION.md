# MMara Backend - Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Multi-Agent System](#multi-agent-system)
4. [API Endpoints](#api-endpoints)
5. [Database Models](#database-models)
6. [Services](#services)
7. [Configuration](#configuration)
8. [Deployment](#deployment)

---

## Overview

**MMara** is an AI-powered legal first-aid assistant designed for Ghanaians. It provides information about Ghanaian law, specifically covering Criminal Law and Road Traffic regulations. The system uses a multi-agent Retrieval-Augmented Generation (RAG) architecture to deliver accurate, contextually relevant legal information.

### Key Features

- **Multi-Agent RAG System**: Orchestrated agents for intake, safety, legal retrieval, and response generation
- **Hybrid Retrieval**: Combines semantic search (Pinecone + OpenAI embeddings) with BM25 keyword search
- **OpenAI Integration**: Uses GPT-4o-mini for LLM completions and text-embedding-3-small for embeddings
- **Ghanaian Legal Documents**: Indexed PDF documents covering criminal and traffic law
- **Real-time Chat**: WebSocket streaming support with conversation history
- **Rate Limiting**: Redis-based rate limiting with role-based quotas
- **JWT Authentication**: Secure authentication with role-based access control (RBAC)

### Technology Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Database | PostgreSQL (async with SQLAlchemy + asyncpg) |
| Cache/Session | Redis |
| Vector DB | Pinecone (serverless, AWS us-east-1) |
| Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| LLM | OpenAI GPT-4o-mini |
| PDF Processing | LlamaParse |
| Reranking | OpenAI LLM-based |
| Tracing | LangSmith |
| Testing | pytest + pytest-asyncio |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                              Client Layer                           │
│                    (Flutter App / Next.js Web Admin)                │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │ JWT Token
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            FastAPI Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │   Auth   │  │   Chat   │  │  Admin   │  │      Users       │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       └────────────────┴──────────┴──────────┘       │               │
│                          │                           │               │
│                   ┌──────▼──────┐            ┌──────▼──────┐        │
│                   │   Security  │            │ Dependency │        │
│                   │   & RBAC    │            │  Injection  │        │
│                   └──────┬──────┘            └──────┬──────┘        │
└──────────────────────────┼──────────────────────────┼───────────────┘
                           │                          │
                           ▼                          ▼
        ┌────────────────────────────────────────────────────┐
        │                 Multi-Agent Orchestrator            │
        │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
        │  │  Intake  │→ │  Safety  │→ │      Legal       │   │
        │  │  Agent   │  │  Agent   │  │      Agent       │   │
        │  └──────────┘  └──────────┘  └────────┬─────────┘   │
        │                                          │            │
        │                              ┌───────────▼────────┐   │
        │                              │  Emergency Handler │   │
        │                              │  (if urgent)       │   │
        │                              └───────────┬────────┘   │
        │                                          │            │
        │                              ┌───────────▼────────┐   │
        │                              │  Response Validator│   │
        │                              └────────────────────┘   │
        └──────────────────────────────────┬───────────────────┘
                                           │
        ┌──────────────────────────────────┼───────────────────┐
        │                                  ▼                   │
        │  ┌──────────────┐      ┌─────────────────┐          │
        │  │ Retrieval    │      │   OpenAI LLM    │          │
        │  │ Service      │─────│   Client        │          │
        │  └──────┬───────┘      └─────────────────┘          │
        │         │                                             │
        │  ┌──────▼───────┐      ┌─────────────────┐          │
        │  │  Embedding   │      │   Redis Cache   │          │
        │  │  Service     │      │   (Sessions)    │          │
        │  └──────┬───────┘      └─────────────────┘          │
        │         │                                             │
        │  ┌──────▼───────┐                                     │
        │  │   Pinecone   │                                     │
        │  │ Vector DB    │                                     │
        │  └──────────────┘                                     │
        └───────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────────────────────┐
        │                   PostgreSQL Database                  │
        │  (Users, Sessions, Documents, Analytics, etc.)        │
        └───────────────────────────────────────────────────────┘
```

### Request Flow

1. **Client Request** → JWT authentication via `Authorization` header
2. **FastAPI Router** → Route to appropriate endpoint
3. **Dependency Injection** → Inject services (DB, Redis, Orchestrator)
4. **Agent Orchestrator** → Multi-agent pipeline execution
5. **Retrieval** → Hybrid search (Pinecone + BM25)
6. **LLM Generation** → OpenAI GPT-4o-mini
7. **Response** → JSON response with citations, confidence, metadata

---

## Multi-Agent System

The multi-agent system is the core of MMara's intelligence. Each agent has a specific responsibility in processing user queries.

### Agent Pipeline

```
Query → IntakeAgent → SafetyAgent → LegalAgent → [EmergencyHandler] → LLM Response → ResponseValidator
```

### Agent Details

#### 1. IntakeAgent (`app/agents/intake.py`)

**Purpose**: Classifies the incoming query to determine how to handle it.

**Classification**:
- **Intent**: `question`, `emergency`, `clarification`, `general`
- **Category**: `criminal`, `road_traffic`, `general`
- **Urgency**: `low`, `medium`, `high`, `critical`
- **Is Emergency**: boolean flag

**Implementation**:
- Quick keyword-based classification for speed
- LLM-based classification override for high-urgency cases
- Language detection (English, Twi support planned)

**Key Methods**:
```python
async def process(context: AgentContext) -> AgentResult:
    # Classifies query and stores metadata in context
    # Returns classification with confidence score
```

#### 2. SafetyAgent (`app/agents/safety.py`)

**Purpose**: Ensures query safety and policy compliance.

**Responsibilities**:
- Filters restricted content (violence, illegal activities, bribery)
- Detects emergency indicators
- Enforces disclaimer requirements
- Validates legal boundaries

**Restricted Topics**:
- Violence, harm, illegal activity
- Evading police, resisting arrest
- Bribery, corruption

**Key Methods**:
```python
async def process(context: AgentContext) -> AgentResult:
    # Returns REJECTED status if unsafe
    # Adds disclaimer and emergency flag to metadata
```

#### 3. LegalAgent (`app/agents/legal.py`)

**Purpose**: Retrieves relevant legal documents and formulates retrieval queries.

**Responsibilities**:
- Formulates optimal retrieval queries
- Performs hybrid search (semantic + BM25)
- Extracts citations from retrieved documents
- Determines if relevant information exists

**Key Methods**:
```python
async def process(context: AgentContext) -> AgentResult:
    # Retrieves documents based on classified category
    # Stores retrieved_documents in context metadata
```

#### 4. EmergencyHandler (`app/agents/safety.py`)

**Purpose**: Provides immediate guidance for urgent situations.

**Activation**: Triggered when `is_emergency=True` from IntakeAgent

**Response Format**:
- Immediate practical guidance
- Summary of rights
- Next steps
- Emergency contacts
- Legal disclaimer

#### 5. ResponseValidator (`app/agents/safety.py`)

**Purpose**: Validates generated responses for quality and safety.

**Validates**:
- Citations are present when appropriate
- Disclaimer is included
- Emergency responses have practical guidance
- Response is not empty or malformed

### Agent Context

The `AgentContext` class carries state between agents:

```python
class AgentContext:
    query: str                    # Original user query
    user_id: Optional[int]        # User identifier
    session_id: str               # Conversation session ID
    conversation_history: List    # Previous messages
    metadata: Dict                # Agent data storage
```

---

## API Endpoints

### Base URL: `/api/v1`

### Authentication Endpoints (`/api/v1/auth`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | Register new user | No |
| POST | `/login` | OAuth2 login (form data) | No |
| POST | `/login/json` | JSON login | No |
| POST | `/refresh` | Refresh access token | No |
| GET | `/me` | Get current user | Yes |
| POST | `/logout` | Logout (client-side) | Yes |
| POST | `/password-reset/request` | Request password reset | No |
| POST | `/password-reset/confirm` | Confirm password reset | No |
| POST | `/change-password` | Change password (authenticated) | Yes |

#### Registration Request
```json
{
  "email": "user@example.com",
  "password": "SecurePass123",
  "full_name": "John Doe",
  "phone": "+233XXXXXXXXX"
}
```

#### Login Response
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

### Chat Endpoints (`/api/v1/chat`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/message` | Send message, get response | Yes |
| GET | `/history` | List user sessions | Yes |
| GET | `/history/{session_id}` | Get specific session | Yes |
| DELETE | `/history/{session_id}` | Delete session | Yes |
| DELETE | `/history` | Delete all sessions | Yes |
| POST | `/feedback` | Submit feedback | Yes |
| WS | `/stream` | WebSocket streaming | Yes |

#### Send Message Request
```json
{
  "message": "What are my rights during arrest?",
  "session_id": "optional-session-id",
  "category": "criminal"
}
```

#### Chat Response
```json
{
  "response": "Based on Ghanaian law...",
  "session_id": "abc123",
  "message_id": "msg456",
  "citations": [{"act": "Act 29", "section": "1", "text": "..."}],
  "confidence": 0.85,
  "category": "criminal",
  "urgency": "medium",
  "is_emergency": false,
  "disclaimer": "I am an AI assistant...",
  "timestamp": "2024-01-15T10:30:00Z",
  "response_time_ms": 1250.5,
  "intent": "question",
  "document_count": 5
}
```

### User Endpoints (`/api/v1/users`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/me` | Get profile | Yes |
| PUT | `/me` | Update profile | Yes |
| GET | `/me/analytics` | Get user analytics | Yes |
| POST | `/me/export` | Export user data | Yes |

### Admin Endpoints (`/api/v1/admin`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/documents` | Upload document | Admin |
| GET | `/documents` | List documents | Admin |
| DELETE | `/documents/{id}` | Delete document | Admin |
| GET | `/stats` | Get system stats | Admin |
| POST | `/reindex` | Reindex documents | Admin |

### Bug Report Endpoints (`/api/v1/bug-reports`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/` | Submit bug report | Yes |
| GET | `/` | List user's reports | Yes |
| GET | `/admin` | List all reports | Admin |
| PUT | `/{id}/status` | Update status | Admin |

---

## Database Models

### Schema Overview

```
users ──┬── chat_sessions ──┐
       ├── analytics ──────┤
       ├── password_resets │
       └── bug_reports ────┘

documents (indexed separately in Pinecone)

rate_limits (Redis-backed, PostgreSQL tracking)
```

### User Model

```python
class User(Base):
    id: int
    email: str                    # Unique, indexed
    hashed_password: str
    full_name: Optional[str]
    phone: Optional[str]
    role: str                     # admin, user, free
    is_active: bool
    is_premium: bool
    created_at: datetime
    updated_at: Optional[datetime]
```

### ChatSession Model

```python
class ChatSession(Base):
    id: int
    session_id: str               # Unique, indexed
    user_id: int                  # FK → users.id
    title: Optional[str]
    category: Optional[str]
    messages: JSON                # List of message objects
    message_count: int
    created_at: datetime
    updated_at: datetime
```

### Document Model

```python
class Document(Base):
    id: int
    filename: str
    original_filename: str
    category: str                 # criminal, road_traffic, general
    doc_type: str                 # pdf, docx
    status: str                   # pending, processing, completed, failed
    chunk_count: int
    file_path: Optional[str]
    doc_metadata: Optional[JSON]
    uploaded_by: Optional[int]    # FK → users.id
    uploaded_at: datetime
    processed_at: Optional[datetime]
```

### Analytics Model

```python
class Analytics(Base):
    id: int
    user_id: Optional[int]        # FK → users.id
    session_id: Optional[str]
    message_id: Optional[str]     # Indexed
    query_type: str
    category: Optional[str]
    urgency: Optional[str]
    response_time_ms: float
    retrieval_count: int
    tokens_used: int
    is_emergency: bool
    satisfaction: Optional[int]   # 1-5 rating
    feedback: Optional[str]
    created_at: datetime          # Indexed

    # Admin fields
    flagged: bool
    flagged_reason: Optional[str]
    admin_response: Optional[str]
    admin_responded_at: Optional[datetime]
    admin_responded_by: Optional[int]
```

### BugReport Model

```python
class BugReport(Base):
    id: int
    user_id: Optional[int]
    session_id: Optional[str]

    # Bug details
    title: str
    description: str
    bug_type: str                 # ui, api, performance, accuracy, other
    severity: str                 # low, medium, high, critical

    # Additional context
    steps_to_reproduce: Optional[str]
    expected_behavior: Optional[str]
    actual_behavior: Optional[str]

    # Device info
    device_info: Optional[str]
    app_version: Optional[str]

    # Admin management
    status: str                   # open, in_progress, resolved, closed
    assigned_to: Optional[int]
    resolution_notes: Optional[str]
    admin_responded_at: Optional[datetime]
```

### PasswordReset Model

```python
class PasswordReset(Base):
    id: int
    user_id: int                  # FK → users.id, indexed
    token: str                    # Unique, indexed
    expires_at: datetime
    used: bool
    created_at: datetime
```

---

## Services

### RetrievalService (`app/services/retrieval.py`)

**Purpose**: Hybrid retrieval combining semantic and keyword search.

**Features**:
- Semantic search via Pinecone embeddings
- BM25 keyword search with rank-bm25
- Reciprocal Rank Fusion (RRF) for result merging
- Optional OpenAI LLM reranking
- BM25 index caching with timeout protection

**Configuration**:
- `alpha`: Semantic search weight (0-1, default 0.7)
- `rrf_k`: RRF constant (default 60)
- `top_k`: Number of results (default 5)

**RRF Formula**:
```
score = alpha * (1 / (k + semantic_rank)) + (1 - alpha) * (1 / (k + keyword_rank))
```

**Key Methods**:
```python
async def retrieve(
    query: str,
    n_results: int = 5,
    filter_metadata: Optional[Dict] = None,
    use_hybrid: bool = True
) -> List[Dict[str, Any]]

async def retrieve_with_rerank(
    query: str,
    n_results: int = 5,
    filter_metadata: Optional[Dict] = None,
    openai_client=None,
    fetch_k: int = 15
) -> List[Dict[str, Any]]
```

### EmbeddingService (`app/services/embeddings.py`)

**Purpose**: Manages vector embeddings and Pinecone integration.

**Configuration**:
- Model: `text-embedding-3-small` (1536 dimensions)
- Index: Pinecone serverless (AWS us-east-1)
- Namespace: `legal_documents`

**Key Methods**:
```python
async def embed_text_async(text: str) -> List[float]
async def embed_batch_async(texts: List[str]) -> List[List[float]]
async def add_chunks(chunks: List[LegalChunk], batch_size: int = 100) -> int
async def search(query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None)
```

### OpenAIClient (`app/services/openai_client.py`)

**Purpose**: Interfaces with OpenAI API for LLM operations.

**Configuration**:
- Model: `gpt-4o-mini` (configurable)
- Temperature: 0.1
- Max tokens: 2000

**Key Methods**:
```python
async def chat(messages: List[Dict]) -> ChatCompletion
async def stream_chat(messages: List[Dict]) -> AsyncGenerator
async def generate_legal_response(query: str, context: List[Dict], conversation_history: Optional[List] = None) -> str
async def classify_query(query: str) -> Dict[str, Any]
async def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]
async def extract_citations(response: str, context: List[Dict]) -> List[Dict]
async def generate_emergency_response(query: str, context: List[Dict]) -> str
```

### RedisService (`app/services/redis_client.py`)

**Purpose**: Manages Redis connections for caching and sessions.

**Uses**:
- Session history storage
- Rate limiting
- Response caching
- BM25 index caching

**Key Methods**:
```python
async def get_session(session_id: str) -> List[Dict]
async def save_session(session_id: str, messages: List[Dict])
async def add_message_to_session(session_id: str, message: Dict)
async def delete_session(session_id: str)
```

### EmailService (`app/services/email_service.py`)

**Purpose**: Sends transactional emails via SMTP.

**Emails**:
- Password reset
- Password change confirmation
- Future: user verification

### LegalDocumentChunker (`app/services/chunker.py`)

**Purpose**: Processes PDF documents into searchable chunks.

**Process**:
1. Extract text from PDF using LlamaParse
2. Split into chunks (512 chars, 50 overlap)
3. Extract metadata (Act, Section, etc.)
4. Return `LegalChunk` objects

---

## Configuration

### Environment Variables

All configuration is managed through `app/config.py` using Pydantic Settings.

**Required Variables**:
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=<your-secret-key>
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

**Optional Variables**:
```bash
# Application
DEBUG=false
ENVIRONMENT=development
APP_NAME=MMara Legal AI
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:19006

# OpenAI
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2000
EMBEDDING_MODEL=text-embedding-3-small

# Pinecone
PINECONE_INDEX_NAME=mmara-legal
PINECONE_NAMESPACE=legal_documents

# Retrieval
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
RETRIEVAL_ALPHA=0.7
RRF_K=60

# Rate Limiting
RATE_LIMIT_FREE=50
RATE_LIMIT_AUTH=500
RATE_LIMIT_PREMIUM=-1

# JWT
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=7

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=...
SMTP_PASSWORD=...
```

### System Prompts

**Legal System Prompt** (`LEGAL_SYSTEM_PROMPT`):
```
You are MMara, an AI-powered legal first-aid assistant for Ghanaians.

Your role is to provide helpful information about Ghanaian law based on the retrieved legal documents.

Guidelines:
1. Use Only Retrieved Context - Base answers ONLY on provided documents
2. Cite Sources - Always cite Acts, Sections, Legislative Instruments
3. Be Clear and Simple - Use plain language
4. Acknowledge Uncertainty - Say so if documents don't contain enough info
5. Emergency Detection - Provide immediate guidance for emergencies
6. Ghanaian Context - Specific to Ghanaian law only
7. Structure Response - Direct answer, legal provisions, practical implications, when to seek help
```

---

## Deployment

### Docker Deployment

**docker-compose.yml** includes:
- PostgreSQL database
- Redis cache
- FastAPI application

```bash
# Start all services
docker-compose up -d

# Initialize database
docker-compose exec app python scripts/seed_db.py

# Process documents
docker-compose exec app python scripts/process_docs.py

# Check logs
docker-compose logs -f app
```

### Production Checklist

1. **Security**
   - [ ] Set strong `SECRET_KEY` (32+ characters)
   - [ ] Set `DEBUG=false`
   - [ ] Configure `CORS_ORIGINS` for production domains
   - [ ] Enable HTTPS/TLS
   - [ ] Use managed PostgreSQL (RDS, Cloud SQL)
   - [ ] Use managed Redis (ElastiCache, Redis Cloud)

2. **API Keys**
   - [ ] OpenAI API key
   - [ ] Pinecone API key
   - [ ] LlamaParse API key (for PDF processing)

3. **Database**
   - [ ] Run migrations (if implemented)
   - [ ] Configure backups
   - [ ] Set connection pooling

4. **Monitoring**
   - [ ] Set up logging aggregation
   - [ ] Configure error tracking (Sentry)
   - [ ] Monitor `/metrics` endpoint
   - [ ] Set up LangSmith tracing

5. **Scaling**
   - [ ] Configure load balancer
   - [ ] Use separate Pinecone instance
   - [ ] Consider read replicas for PostgreSQL
   - [ ] Scale Redis horizontally

### Health Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /info` - Application info
- `GET /metrics` - Metrics (debug only)

---

## Error Handling

### Error Responses

All errors follow this format:

```json
{
  "error": "Error message",
  "status_code": 400,
  "request_id": "uuid-of-request"
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request / Validation Error |
| 401 | Unauthorized / Invalid token |
| 403 | Forbidden / Insufficient permissions |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Rate Limiting

### Rate Limits by Role

| Role | Requests/Day |
|------|--------------|
| Free | 50 |
| User | 500 |
| Premium | Unlimited (-1) |

### Implementation

- Redis-backed rate limiting
- Sliding window algorithm
- Per-IP and per-user tracking
- Configurable per endpoint

---

## WebSocket Streaming

### Endpoint: `WS /api/v1/chat/stream`

### Connection Flow

1. Send initial message with auth:
```json
{
  "token": "jwt-token",
  "message": "What are my rights?",
  "session_id": "optional"
}
```

2. Receive streaming chunks:
```json
{"type": "status", "data": {"stage": "classifying", "message": "Analyzing query..."}}
{"type": "classification", "data": {"intent": "question", "category": "criminal"}}
{"type": "status", "data": {"stage": "retrieving", "message": "Searching documents..."}}
{"type": "retrieval", "data": {"count": 5}}
{"type": "status", "data": {"stage": "generating", "message": "Generating response..."}}
{"type": "token", "data": {"content": "Based"}}
{"type": "token", "data": {"content": " on"}}
{"type": "complete", "data": {"response": "...", "citations": [], "session_id": "..."}}
```

---

## Document Processing

### Process Script: `scripts/process_docs.py`

**Usage**:
```bash
# Process all configured directories
python scripts/process_docs.py

# Process specific directory
python scripts/process_docs.py -d /path/to/pdfs -c criminal

# Rebuild database first
python scripts/process_docs.py --rebuild
```

**Process**:
1. Find all PDF files in configured directories
2. Extract text using LlamaParse
3. Split into chunks with metadata
4. Generate embeddings using OpenAI
5. Store in Pinecone vector database
6. Track in PostgreSQL documents table

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_agents.py -v

# Specific test
pytest tests/test_agents.py::test_intake_agent -v
```

### Test Structure

```
tests/
├── conftest.py           # Fixtures
├── test_agents.py        # Agent tests
├── test_api.py           # API endpoint tests
├── test_retrieval.py     # Retrieval service tests
└── test_services.py      # Other service tests
```

---

## Security Features

1. **Authentication**
   - JWT Bearer tokens (24hr access, 7day refresh)
   - Password hashing with bcrypt
   - Password requirements: 8+ chars, uppercase, lowercase, digit

2. **Authorization**
   - Role-based access control (admin, user, free)
   - Protected endpoints with `@Depends(get_current_user)`

3. **Rate Limiting**
   - Redis-based per-user and per-IP limits
   - Strict limits on auth endpoints

4. **Security Headers**
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - Content-Security-Policy
   - Strict-Transport-Security (production)

5. **Input Validation**
   - Pydantic models for all inputs
   - SQL injection prevention (parameterized queries)
   - XSS prevention

---

## Monitoring & Observability

### Logging

- JSON structured logging
- Request ID tracking
- Agent pipeline logging
- Performance timing

### Metrics

- Request count and duration
- Agent execution times
- Retrieval statistics
- Error tracking

### Tracing

- LangSmith integration for LLM calls
- Request/response tracking
- Agent pipeline tracing

---

## License

[Your License Here]
