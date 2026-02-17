"""
Admin API endpoints.
Handles document management and system administration.
"""

import time
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
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
from app.models.user import UserInDB
from app.services.chunker import LegalDocumentChunker

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    current_user: Annotated[UserInDB, Depends(AdminUser)],
    db: DBSession,
    embedding_service: EmbeddingSvc,
    file: UploadFile = File(...),
    category: str = "general",
    doc_type: str = "other",
):
    """
    Upload a legal document PDF to be processed and indexed.

    - **file**: PDF file to upload
    - **category**: Document category (criminal, road_traffic, general)
    - **doc_type**: Document type (act, amendment, regulation, legislative_instrument, other)

    Requires admin privileges.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Create documents directory if needed
    upload_dir = embedding_service.persist_directory / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = upload_dir / safe_filename

    with open(file_path, "wb") as f:
        content = await file.read()
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
    current_user: Annotated[UserInDB, Depends(AdminUser)],
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
    current_user: Annotated[UserInDB, Depends(AdminUser)],
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

    # Note: Chunks in ChromaDB would need to be deleted separately
    # by filtering on source_file metadata

    return {"message": "Document deleted successfully"}


@router.post("/reindex")
async def reindex_documents(
    request: ReindexRequest,
    current_user: Annotated[UserInDB, Depends(AdminUser)],
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
    current_user: Annotated[UserInDB, Depends(AdminUser)],
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
    current_user: Annotated[UserInDB, Depends(AdminUser)],
    db: DBSession,
    embedding_service: EmbeddingSvc
):
    """
    Get statistics about the document collection.

    Requires admin privileges.
    """
    # Get ChromaDB stats
    chroma_stats = embedding_service.get_collection_stats()

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
        total_chunks=chroma_stats["total_documents"],
        by_category=category_stats,
        by_doc_type=doc_type_stats,
        last_updated=last_updated or time.time()
    )


@router.post("/process-pending")
async def process_pending_documents(
    current_user: Annotated[UserInDB, Depends(AdminUser)],
    db: DBSession,
    embedding_service: EmbeddingSvc
):
    """
    Process all pending documents.

    Requires admin privileges.
    """
    # Get pending documents
    result = await db.execute(
        select(Document).where(Document.status == "pending")
    )
    documents = result.scalars().all()

    processed = 0
    for document in documents:
        try:
            file_path = Path(document.file_path) if document.file_path else None
            if file_path and file_path.exists():
                # Process the document
                chunker = LegalDocumentChunker(
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap
                )
                chunks = await chunker.chunk_document_async(file_path)

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
            document.status = "failed"
            document.metadata = document.metadata or {}
            document.metadata["error"] = str(e)

    await db.commit()

    return {
        "message": f"Processed {processed} of {len(documents)} pending documents",
        "processed": processed,
        "total": len(documents)
    }
