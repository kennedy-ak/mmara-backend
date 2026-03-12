"""
Document-related Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a legal document."""

    source_file: str
    total_pages: Optional[int] = None
    act_number: Optional[str] = None
    year: Optional[str] = None
    legislative_instrument: Optional[str] = None
    doc_type: Optional[str] = None
    category: Optional[str] = None
    section_header: Optional[str] = None
    section_number: Optional[int] = None


class DocumentChunk(BaseModel):
    """A chunk of legal text."""

    chunk_id: str
    text: str
    metadata: DocumentMetadata


class DocumentUpload(BaseModel):
    """Model for document upload request."""

    filename: str
    category: str = Field(..., pattern="^(criminal|road_traffic|general)$")
    doc_type: str = Field(..., pattern="^(act|amendment|regulation|legislative_instrument|other)$")


class DocumentInfo(BaseModel):
    """Information about a processed document."""

    id: int
    filename: str
    category: str
    doc_type: str
    status: str
    chunk_count: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None


class RetrievalResult(BaseModel):
    """Result from retrieval service."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    rerank_score: Optional[float] = None


class RetrievalRequest(BaseModel):
    """Request for retrieval."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = Field(None, pattern="^(criminal|road_traffic|general)$")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    rerank: bool = True


class ReindexRequest(BaseModel):
    """Request to reindex documents."""

    categories: Optional[List[str]] = Field(None, pattern="^(criminal|road_traffic|general)$")
    force: bool = False


class DocumentStats(BaseModel):
    """Statistics about document collection."""

    total_documents: int
    total_chunks: int
    by_category: Dict[str, int]
    by_doc_type: Dict[str, int]
    last_updated: Optional[datetime] = None
