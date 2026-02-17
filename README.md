# MMara Backend

AI-powered legal first-aid assistant for Ghanaians. Built with FastAPI, implementing a multi-agent RAG system for legal information retrieval.

## Features

- **Multi-Agent RAG System**: Intake, Legal, Safety, and Orchestrator agents working together
- **Hybrid Retrieval**: Combines semantic search (ChromaDB + sentence-transformers) with BM25 keyword search
- **Groq LLM Integration**: Fast inference using Llama 3.1 70B
- **Ghanaian Legal Documents**: Criminal Law, Road Traffic Acts, and more
- **Real-time Chat**: With conversation history and context awareness
- **Rate Limiting**: Redis-based rate limiting for API endpoints
- **JWT Authentication**: Secure user authentication with role-based access control

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Database | PostgreSQL (async with SQLAlchemy) |
| Cache/Session | Redis |
| Vector DB | ChromaDB (embedded) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| LLM | Groq (Llama 3.1 70B) |
| Container | Docker |

## Project Structure

```
mmara-backend/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── agents/          # Multi-agent system
│   ├── core/            # Security, RBAC, rate limiting
│   ├── db/              # Database models and session
│   ├── models/          # Pydantic models
│   ├── services/        # Business logic services
│   ├── utils/           # Logging, metrics
│   ├── config.py        # Configuration
│   ├── dependencies.py  # Dependency injection
│   └── main.py          # FastAPI app entry
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
# Copy environment file
cp .env.example .env

# Edit .env and set your variables
# Required: GROQ_API_KEY
```

### 2. Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Process legal documents
docker-compose exec app python scripts/process_docs.py

# Seed database with default users
docker-compose exec app python scripts/seed_db.py
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL and Redis (or use docker-compose for those only)
docker-compose up -d postgres redis

# Initialize database
python scripts/seed_db.py

# Process documents
python scripts/process_docs.py

# Run server
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

## Default Credentials

After running `seed_db.py`:

| User | Email | Password | Role |
|------|-------|----------|------|
| Admin | admin@mmara.gh | Admin123! | admin |
| Test | test@mmara.gh | Test123! | user |

**Important:** Change these passwords in production!

## Scripts

### `scripts/process_docs.py`
Process legal PDFs into the vector database.

```bash
# Process all configured directories
python scripts/process_docs.py

# Process specific directory
python scripts/process_docs.py -d /path/to/pdfs -c criminal

# Rebuild database first
python scripts/process_docs.py --rebuild
```

### `scripts/seed_db.py`
Initialize database with default users.

```bash
python scripts/seed_db.py
```

### `scripts/test_system.py`
Test all system components.

```bash
python scripts/test_system.py
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `GROQ_API_KEY` | Groq API key (required) | - |
| `SECRET_KEY` | JWT secret key | `change-this...` |
| `CHROMA_PERSIST_DIR` | Vector database path | `./data/chroma_db` |
| `CRIMINAL_DOCS_PATH` | Criminal law documents | `../Criminal_proceeding` |
| `ROAD_TRAFFIC_DOCS_PATH` | Traffic law documents | `../Road_traffic` |

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get token
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user

### Chat
- `POST /api/v1/chat/message` - Send a message
- `GET /api/v1/chat/history` - Get chat sessions
- `GET /api/v1/chat/history/{id}` - Get specific session
- `DELETE /api/v1/chat/history/{id}` - Delete session
- `WS /api/v1/chat/stream` - WebSocket streaming

### Admin (requires admin role)
- `POST /api/v1/admin/documents` - Upload document
- `GET /api/v1/admin/documents` - List documents
- `DELETE /api/v1/admin/documents/{id}` - Delete document
- `GET /api/v1/admin/stats` - Get system stats
- `POST /api/v1/admin/reindex` - Reindex documents

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v
```

## Deployment

### Production Checklist

1. **Security**
   - [ ] Change `SECRET_KEY` to a strong random value
   - [ ] Set `DEBUG=false`
   - [ ] Configure `CORS_ORIGINS` appropriately
   - [ ] Set up SSL/TLS certificates

2. **Database**
   - [ ] Use managed PostgreSQL (RDS, Cloud SQL, etc.)
   - [ ] Configure backups
   - [ ] Run migrations

3. **Monitoring**
   - [ ] Set up logging aggregation
   - [ ] Configure alerting
   - [ ] Monitor `/metrics` endpoint

4. **Scaling**
   - [ ] Use managed Redis (ElastiCache, etc.)
   - [ ] Consider separate ChromaDB instance
   - [ ] Configure load balancer

## License

[Your License Here]
