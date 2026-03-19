"""
Configuration module for MMara Backend.
Uses Pydantic Settings for type-safe configuration.
"""

from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def parse_cors_origins(v: Any) -> List[str]:
    """Parse CORS origins from comma-separated string or list."""
    if isinstance(v, str):
        return [origin.strip() for origin in v.split(",")]
    return v


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
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
        default=["http://localhost:3000", "http://localhost:19006"],
        validate_default=True,
    )

    # Database - PostgreSQL
    database_url: str = Field(
        default="", alias="DATABASE_URL"
    )
    database_echo: bool = False

    # Redis
    # Formats:
    # - redis://localhost:6379/0 (no password)
    # - redis://:password@localhost:6379/0 (with password)
    # - rediss://:password@localhost:6379/0 (with password and SSL)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_ttl: int = 3600  # 1 hour
    redis_max_connections: int = Field(default=50, alias="REDIS_MAX_CONNECTIONS")

    # JWT Authentication
    secret_key: str = Field(default="dev-only-change-in-production-not-secure", alias="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    refresh_token_expire_days: int = 7

    # OpenAI API
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2000

    # Embeddings (OpenAI)
    embedding_model: str = "text-embedding-3-small"  # 1536-dim

    # LlamaParse
    llamaparse_api_key: Optional[str] = Field(None, alias="LLAMAPARSE_API_KEY")

    # Pinecone
    pinecone_api_key: Optional[str] = Field(None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="mmara-legal", alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="legal_documents", alias="PINECONE_NAMESPACE")

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

    # Observo Logging
    observo_api_key: Optional[str] = Field(None, alias="OBSERVO_API_KEY")
    observo_project_id: Optional[str] = Field(None, alias="OBSERVO_PROJECT_ID")
    observo_url: str = Field(
        default="https://observo-log.vendlyghana.space/api/v1/ingest/",
        alias="OBSERVO_URL",
    )

    # Email Configuration
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    smtp_host: Optional[str] = Field(None, alias="SMTP_HOST")
    smtp_port: int = Field(587, alias="SMTP_PORT")
    smtp_use_tls: bool = Field(True, alias="SMTP_USE_TLS")
    smtp_username: Optional[str] = Field(None, alias="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, alias="SMTP_PASSWORD")
    smtp_from_email: Optional[str] = Field(None, alias="SMTP_FROM_EMAIL")
    smtp_from_name: str = Field("MMara Legal AI", alias="SMTP_FROM_NAME")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> Any:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        """Validate that critical security settings are properly configured."""
        errors = []

        # In production, require secure credentials
        if self.environment == "production":
            if not self.secret_key or len(self.secret_key) < 32:
                errors.append(
                    "SECRET_KEY must be set and at least 32 characters in production"
                )
            if not self.database_url:
                errors.append("DATABASE_URL must be set in production")

            # Check for obvious placeholder values
            placeholder_keys = [
                "change-this-secret-key",
                "your-secret-key",
                "replace-with-secret",
            ]
            if any(placeholder in self.secret_key.lower() for placeholder in placeholder_keys):
                errors.append("SECRET_KEY appears to be a placeholder. Set a secure random key.")

        if errors:
            raise ValueError(f"Security configuration errors: {', '.join(errors)}")

        return self

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

3. **Be Clear and Simple**: Use plain language that non-lawyers can understand. Explain legal terms when necessary.

4. **Acknowledge Uncertainty**: If the retrieved documents don't contain enough information to answer the question, say so explicitly. Do not speculate.

5. **Emergency Detection**: If the user indicates an emergency (e.g., "I'm being arrested right now"), provide immediate practical guidance and recommend contacting a lawyer.

6. **Ghanaian Context**: Remember that your advice is specific to Ghanaian law. Do not reference laws from other jurisdictions.

7. **Structure Your Response**:
   - Direct answer to the question
   - Relevant legal provisions with citations
   - Practical implications (only when relevant to the user's situation)
   - When to seek professional legal help

Do NOT include a disclaimer in your response. The application already displays a legal disclaimer to the user."""

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
