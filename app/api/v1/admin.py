"""
Admin API endpoints.
Handles document management and system administration.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Set

logger = logging.getLogger("mmara")

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.config import settings
from app.db.models import Document
from app.dependencies import AdminUser, DBSession, EmbeddingSvc, RetrievalSvc
from app.models.document import (
    DocumentInfo,
    DocumentStats,
    ReindexRequest,
    RetrievalRequest,
    RetrievalResult,
)
from app.services.chunker import LegalDocumentChunker

router = APIRouter(prefix="/admin", tags=["Admin"])

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_CATEGORIES: Set[str] = {"criminal", "road_traffic", "general"}
ALLOWED_DOC_TYPES: Set[str] = {"act", "amendment", "regulation", "legislative_instrument", "other"}
ALLOWED_EXTENSIONS = {".pdf"}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        str: Sanitized safe filename
    """
    # Remove any path components
    filename = Path(filename).name

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Replace any characters that aren't alphanumeric, dash, underscore, or dot
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Ensure filename isn't empty after sanitization
    if not filename:
        filename = "unnamed_file.pdf"

    # Limit filename length
    name, ext = Path(filename).stem, Path(filename).suffix
    if len(name) > 100:
        name = name[:100]
    filename = f"{name}{ext}"

    return filename


def validate_pdf_content(content: bytes) -> bool:
    """
    Validate that the file content is actually a PDF.

    Args:
        content: File content as bytes

    Returns:
        bool: True if content appears to be a valid PDF
    """
    # PDF files should start with %PDF- (magic number)
    if len(content) < 5:
        return False
    return content[:4] == b"%PDF"


@router.post(
    "/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    current_user: AdminUser,
    db: DBSession,
    embedding_service: EmbeddingSvc,
    file: UploadFile = File(...),
    category: str = Form("general"),
    doc_type: str = Form("other"),
):
    """
    Upload a legal document PDF to be processed and indexed.

    - **file**: PDF file to upload (max 50MB)
    - **category**: Document category (criminal, road_traffic, general)
    - **doc_type**: Document type (act, amendment, regulation, legislative_instrument, other)

    Requires admin privileges.
    """
    # Validate category
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category. Must be one of: {', '.join(ALLOWED_CATEGORIES)}"
        )

    # Validate doc_type
    if doc_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid doc_type. Must be one of: {', '.join(ALLOWED_DOC_TYPES)}"
        )

    # Validate file exists and has name
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Got: {file_ext}"
        )

    # Read and validate file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )

    # Validate PDF magic number
    if not validate_pdf_content(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file. File does not appear to be a valid PDF."
        )

    # Create documents directory if needed
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_original_filename = sanitize_filename(file.filename)
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{safe_original_filename}"
    file_path = upload_dir / safe_filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    # Create database entry
    document = Document(
        filename=safe_filename,
        original_filename=file.filename,
        category=category,
        doc_type=doc_type,
        status="processing",
        file_path=str(file_path),
        uploaded_by=current_user.id
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Process document asynchronously (in background)
    # For now, we'll mark it as pending
    document.status = "pending"
    await db.commit()

    return DocumentInfo(
        id=document.id,
        filename=document.original_filename,
        category=document.category,
        doc_type=document.doc_type,
        status=document.status,
        chunk_count=0,
        uploaded_at=document.uploaded_at
    )


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    current_user: AdminUser,
    db: DBSession,
    skip: int = 0,
    limit: int = 50
):
    """
    List all processed documents.

    - **skip**: Number of documents to skip
    - **limit**: Maximum number of documents to return

    Requires admin privileges.
    """
    result = await db.execute(
        select(Document)
        .order_by(Document.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()

    return [
        DocumentInfo(
            id=doc.id,
            filename=doc.original_filename,
            category=doc.category,
            doc_type=doc.doc_type,
            status=doc.status,
            chunk_count=doc.chunk_count,
            uploaded_at=doc.uploaded_at,
            processed_at=doc.processed_at
        )
        for doc in documents
    ]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: AdminUser,
    db: DBSession,
    embedding_service: EmbeddingSvc
):
    """
    Delete a document and its associated chunks.

    Requires admin privileges.
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Delete file
    if document.file_path and Path(document.file_path).exists():
        Path(document.file_path).unlink()

    # Delete from database
    await db.delete(document)
    await db.commit()

    # Delete associated chunks from Pinecone by source_file metadata
    try:
        embedding_service.index.delete(
            filter={"source_file": document.original_filename},
            namespace=embedding_service.namespace,
        )
    except Exception:
        pass  # Best-effort deletion; document row is already removed

    return {"message": "Document deleted successfully"}


@router.post("/reindex")
async def reindex_documents(
    request: ReindexRequest,
    current_user: AdminUser,
    embedding_service: EmbeddingSvc
):
    """
    Rebuild the vector index from scratch or for specific categories.

    - **categories**: Optional list of categories to reindex
    - **force**: Whether to force reindexing even if index exists

    Requires admin privileges.
    """
    if request.force:
        # Rebuild entire collection
        embedding_service.rebuild_collection()

    return {
        "message": "Reindex initiated. Use the processing script to complete.",
        "categories": request.categories
    }


@router.post("/retrieve", response_model=List[RetrievalResult])
async def test_retrieval(
    request: RetrievalRequest,
    current_user: AdminUser,
    retrieval_service: RetrievalSvc
):
    """
    Test the retrieval system with a custom query.

    - **query**: Search query
    - **top_k**: Number of results to return
    - **category**: Optional category filter
    - **alpha**: Semantic vs keyword weight (0-1)
    - **rerank**: Whether to apply reranking

    Requires admin privileges.
    """
    filter_metadata = {"category": request.category} if request.category else None

    results = await retrieval_service.retrieve(
        query=request.query,
        n_results=request.top_k,
        filter_metadata=filter_metadata
    )

    return [
        RetrievalResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            metadata=r["metadata"],
            score=r.get("rrf_score", 0)
        )
        for r in results
    ]


@router.get("/stats", response_model=DocumentStats)
async def get_document_stats(
    current_user: AdminUser,
    db: DBSession,
    embedding_service: EmbeddingSvc
):
    """
    Get statistics about the document collection.

    Requires admin privileges.
    """
    # Get vector DB stats
    vector_stats = embedding_service.get_collection_stats()

    # Get database stats
    doc_count_result = await db.execute(
        select(func.count(Document.id))
    )
    total_docs = doc_count_result.scalar() or 0

    # Get stats by category
    category_stats = {}
    for category in ["criminal", "road_traffic", "general"]:
        result = await db.execute(
            select(func.count(Document.id))
            .where(Document.category == category)
        )
        category_stats[category] = result.scalar() or 0

    # Get stats by doc type
    doc_type_stats = {}
    for doc_type in ["act", "amendment", "regulation", "legislative_instrument", "other"]:
        result = await db.execute(
            select(func.count(Document.id))
            .where(Document.doc_type == doc_type)
        )
        doc_type_stats[doc_type] = result.scalar() or 0

    # Get last updated time
    last_updated_result = await db.execute(
        select(Document.processed_at)
        .where(Document.processed_at.isnot(None))
        .order_by(Document.processed_at.desc())
        .limit(1)
    )
    last_updated = last_updated_result.scalar_one_or_none()

    return DocumentStats(
        total_documents=total_docs,
        total_chunks=vector_stats["total_documents"],
        by_category=category_stats,
        by_doc_type=doc_type_stats,
        last_updated=last_updated
    )


@router.post("/process-pending")
async def process_pending_documents(
    current_user: AdminUser,
    db: DBSession,
    embedding_service: EmbeddingSvc,
    retrieval_service: RetrievalSvc,
):
    """
    Process all pending documents.

    Requires admin privileges.
    """
    # Get pending and failed documents (retry failed ones too)
    result = await db.execute(
        select(Document).where(Document.status.in_(["pending", "failed"]))
    )
    documents = result.scalars().all()

    processed = 0
    errors = []
    for document in documents:
        try:
            file_path = Path(document.file_path) if document.file_path else None
            if not file_path or not file_path.exists():
                error_msg = f"File not found: {document.file_path}"
                logger.error(f"Document {document.id} ({document.original_filename}): {error_msg}")
                document.status = "failed"
                document.doc_metadata = document.doc_metadata or {}
                document.doc_metadata["error"] = error_msg
                errors.append({"id": document.id, "filename": document.original_filename, "error": error_msg})
                continue

            logger.info(f"Processing document {document.id}: {document.original_filename}")
            document.status = "processing"
            await db.commit()

            # Process the document
            chunker = LegalDocumentChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            chunks = await chunker.chunk_document_async(file_path)

            logger.info(f"Document {document.id}: extracted {len(chunks)} chunks")

            # Add category to chunks
            for chunk in chunks:
                chunk.metadata["category"] = document.category

            # Add to embedding service
            await embedding_service.add_chunks(chunks)

            # Update document status
            document.status = "processed"
            document.chunk_count = len(chunks)
            document.processed_at = func.now()
            processed += 1

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Document {document.id} ({document.original_filename}) failed: {error_msg}", exc_info=True)
            document.status = "failed"
            document.doc_metadata = document.doc_metadata or {}
            document.doc_metadata["error"] = error_msg
            errors.append({"id": document.id, "filename": document.original_filename, "error": error_msg})

    await db.commit()

    # Invalidate BM25 index so it rebuilds with new documents on next search
    if processed > 0:
        retrieval_service.invalidate_bm25_index()

    return {
        "message": f"Processed {processed} of {len(documents)} documents",
        "processed": processed,
        "total": len(documents),
        "errors": errors,
    }
