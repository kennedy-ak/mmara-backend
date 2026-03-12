"""
Document Processing Script
Process legal PDF documents and add them to the vector database.
"""

import argparse
import asyncio
from pathlib import Path

from app.config import settings
from app.services.chunker import LegalDocumentChunker
from app.services.embeddings import EmbeddingService


async def process_documents(directory: str, category: str = None, force_rebuild: bool = False):
    """
    Process all PDFs in a directory.

    Args:
        directory: Directory containing PDFs
        category: Optional category to tag documents with
        force_rebuild: Whether to rebuild the collection
    """
    print("=" * 60)
    print("MMara Document Processor")
    print("=" * 60)

    # Initialize services
    embedding_service = EmbeddingService(
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        embedding_model=settings.embedding_model,
    )

    # Rebuild collection if requested
    if force_rebuild:
        print("\nRebuilding collection...")
        embedding_service.rebuild_collection()

    # Initialize chunker
    chunker = LegalDocumentChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    # Process directory
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return

    print(f"\nProcessing directory: {directory}")
    print(f"Category: {category or 'auto-detect'}")

    # Chunk documents
    chunks_by_file = await chunker.chunk_directory(dir_path, category=category)

    if not chunks_by_file:
        print("No documents found or processed.")
        return

    # Flatten chunks
    all_chunks = []
    for file_path, chunks in chunks_by_file.items():
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to process: {len(all_chunks)}")

    # Add to vector database
    print("\nAdding chunks to vector database...")
    count = await embedding_service.add_chunks(all_chunks)

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"Chunks added: {count}")
    print("=" * 60)

    # Show stats
    stats = embedding_service.get_collection_stats()
    print("\nCollection Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding model: {stats['embedding_model']}")


async def process_all_documents(force_rebuild: bool = False):
    """Process all configured document directories."""
    criminal_path = settings.criminal_docs_path_resolved
    traffic_path = settings.road_traffic_docs_path_resolved

    print("\n=== Processing Criminal Law Documents ===")
    if criminal_path.exists():
        await process_documents(str(criminal_path), "criminal", force_rebuild)
    else:
        print(f"Path not found: {criminal_path}")

    print("\n=== Processing Road Traffic Documents ===")
    if traffic_path.exists():
        await process_documents(str(traffic_path), "road_traffic", force_rebuild)
    else:
        print(f"Path not found: {traffic_path}")


async def main():
    parser = argparse.ArgumentParser(description="Process legal documents for MMara")
    parser.add_argument(
        "--directory",
        "-d",
        help="Specific directory to process (default: all configured directories)",
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=["criminal", "road_traffic", "general"],
        help="Category to tag documents with",
    )
    parser.add_argument(
        "--rebuild", "-r", action="store_true", help="Rebuild the vector database before processing"
    )

    args = parser.parse_args()

    if args.directory:
        await process_documents(args.directory, args.category, args.rebuild)
    else:
        await process_all_documents(args.rebuild)


if __name__ == "__main__":
    asyncio.run(main())
