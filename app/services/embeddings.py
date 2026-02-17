"""
Embeddings Service for Legal Documents.
Handles vector embeddings using sentence-transformers and ChromaDB.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.chunker import LegalChunk


class EmbeddingService:
    """
    Manage embeddings and vector database for legal documents.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "legal_documents",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or Path("./data/chroma_db")
        self.embedding_model_name = embedding_model

        # Initialize embedding model (load in background to avoid blocking)
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim = 384  # Default for all-MiniLM-L6-v2

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.embedding_model_name, device=settings.embedding_device
            )
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, show_progress_bar=True).tolist()

    async def embed_text_async(self, text: str) -> List[float]:
        """Async wrapper for embedding single text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper for embedding batch."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)

    async def add_chunks(self, chunks: List[LegalChunk], batch_size: int = 100) -> int:
        """
        Add document chunks to the vector database.

        Args:
            chunks: List of LegalChunk objects
            batch_size: Number of chunks to process at once

        Returns:
            Number of chunks added
        """
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            # Generate embeddings
            embeddings = await self.embed_batch_async(texts)

            # Add to collection
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )

            print(f"  Added {i + len(batch)}/{total_chunks} chunks")

        return total_chunks

    async def search(
        self, query: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            Dictionary with search results
        """
        query_embedding = await self.embed_text_async(query)

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=filter_metadata
        )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()

        # Get a sample to infer metadata structure
        sample = self.collection.get(limit=1)

        return {
            "name": self.collection_name,
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "has_data": count > 0,
            "sample_metadata_keys": (
                list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
            ),
        }

    def delete_collection(self):
        """Delete the current collection."""
        self.client.delete_collection(self.collection_name)

    def rebuild_collection(self):
        """Delete and recreate the collection."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            List of document dictionaries
        """
        result = self.collection.get(ids=ids, include=["documents", "metadatas"])

        documents = []
        for i, doc_id in enumerate(result["ids"]):
            documents.append(
                {"id": doc_id, "text": result["documents"][i], "metadata": result["metadatas"][i]}
            )

        return documents
