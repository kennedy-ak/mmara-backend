"""
Configuration module for MMara Backend.
Uses Pydantic Settings for type-safe configuration.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application
    app_name: str = "MMara Legal AI"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:19006"], alias="CORS_ORIGINS"
    )

    # Database - PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://mmara:mmara@localhost:5432/mmara", alias="DATABASE_URL"
    )
    database_echo: bool = False

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_ttl: int = 3600  # 1 hour

    # JWT Authentication
    secret_key: str = Field(default="change-this-secret-key-in-production", alias="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    refresh_token_expire_days: int = 7

    # Groq API
    groq_api_key: Optional[str] = Field(None, alias="GROQ_API_KEY")
    groq_model: str = "llama-3.1-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2000

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda" if available

    # ChromaDB
    chroma_persist_directory: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = "legal_documents"

    # Document Paths
    criminal_docs_path: str = Field(default="../Criminal_proceeding", alias="CRIMINAL_DOCS_PATH")
    road_traffic_docs_path: str = Field(default="../Road_traffic", alias="ROAD_TRAFFIC_DOCS_PATH")

    # Retrieval Parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    retrieval_alpha: float = 0.7  # Semantic vs keyword weight
    rrf_k: int = 60  # Reciprocal rank fusion constant

    # Rate Limiting
    rate_limit_free: int = 50  # requests per day
    rate_limit_auth: int = 500  # requests per day
    rate_limit_premium: int = -1  # unlimited

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "text"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"

    @property
    def criminal_docs_path_resolved(self) -> Path:
        """Get the resolved criminal documents path."""
        path = Path(self.criminal_docs_path)
        return path if path.is_absolute() else self.project_root.parent / path

    @property
    def road_traffic_docs_path_resolved(self) -> Path:
        """Get the resolved road traffic documents path."""
        path = Path(self.road_traffic_docs_path)
        return path if path.is_absolute() else self.project_root.parent / path

    @property
    def chroma_persist_directory_resolved(self) -> Path:
        """Get the resolved ChromaDB persist directory."""
        path = Path(self.chroma_persist_directory)
        return path if path.is_absolute() else self.project_root / path


# Global settings instance
settings = Settings()


# Legal categories configuration
LEGAL_CATEGORIES = {
    "criminal": {
        "description": "Ghanaian Criminal Law and Procedure",
        "topics": ["arrest", "bail", "detention", "search", "seizure", "rights"],
    },
    "road_traffic": {
        "description": "Ghanaian Road Traffic Laws and Regulations",
        "topics": ["traffic stop", "dangerous driving", "license", "insurance", "accident"],
    },
}

# System prompts
LEGAL_SYSTEM_PROMPT = """You are MMara, an AI-powered legal first-aid assistant for Ghanaians.

Your role is to provide helpful information about Ghanaian law based on the retrieved legal documents. Follow these guidelines:

1. **Use Only Retrieved Context**: Base your answers ONLY on the legal documents provided in the context. Do not make up or guess legal information.

2. **Cite Sources**: Always cite specific Acts, Sections, and Legislative Instruments when referencing legal provisions. Use the format: "Act 29, Section 1" or "Road Traffic Act, 2004 (Act 683), Section 5"

3. **Include Disclaimer**: Every response must include the following disclaimer: "I am an AI assistant, not a qualified lawyer. This information is for educational purposes only and does not constitute legal advice. For serious legal matters, please consult a qualified lawyer."

4. **Be Clear and Simple**: Use plain language that non-lawyers can understand. Explain legal terms when necessary.

5. **Acknowledge Uncertainty**: If the retrieved documents don't contain enough information to answer the question, say so explicitly. Do not speculate.

6. **Emergency Detection**: If the user indicates an emergency (e.g., "I'm being arrested right now"), provide immediate practical guidance and recommend contacting a lawyer.

7. **Ghanaian Context**: Remember that your advice is specific to Ghanaian law. Do not reference laws from other jurisdictions.

8. **Structure Your Response**:
   - Direct answer to the question
   - Relevant legal provisions with citations
   - Practical implications
   - When to seek professional legal help
   - Disclaimer"""

EMERGENCY_RESPONSE_TEMPLATE = """This sounds like an urgent situation. Here's what you should know right now:

{immediate_guidance}

**Important Rights:**
{rights_summary}

**Next Steps:**
1. Remain calm and polite
2. Do not resist, but clearly state your rights
3. Contact a lawyer as soon as possible
4. If possible, have someone witness the interaction

**Emergency Contacts:**
- Police Complaints Unit: [Add local number]
- Legal Aid: [Add local number]

I am an AI assistant, not a qualified lawyer. This is general information, not legal advice."""


class QueryCategory:
    """Query classification categories."""

    QUESTION = "question"
    EMERGENCY = "emergency"
    CLARIFICATION = "clarification"
    GENERAL = "general"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.QUESTION, cls.EMERGENCY, cls.CLARIFICATION, cls.GENERAL]


class LegalCategory:
    """Legal document categories."""

    CRIMINAL = "criminal"
    ROAD_TRAFFIC = "road_traffic"
    GENERAL = "general"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.CRIMINAL, cls.ROAD_TRAFFIC, cls.GENERAL]


class UrgencyLevel:
    """Query urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
