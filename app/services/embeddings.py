"""
Embeddings Service for Legal Documents.
Handles vector embeddings using OpenAI and Pinecone.
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.services.chunker import LegalChunk


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata values to Pinecone-compatible types (str/int/float/bool/list[str])."""
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [str(v) for v in value]
        else:
            sanitized[key] = str(value)
    return sanitized


class EmbeddingService:
    """
    Manage embeddings and vector database for legal documents.
    Uses Pinecone as the vector store.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "mmara-legal",
        namespace: str = "legal_documents",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = embedding_model
        self._embedding_dim = 1536  # OpenAI text-embedding-3-small dimension

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key or settings.openai_api_key)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Auto-create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=self._embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(index_name)

    async def embed_text_async(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        response = await self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model_name,
        )
        return response.data[0].embedding

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        response = await self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model_name,
        )
        # Sort by index to preserve order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    async def add_chunks(self, chunks: List[LegalChunk], batch_size: int = 100) -> int:
        """
        Add document chunks to Pinecone.

        Stores chunk text inside metadata["text"] (well within 40KB limit at ~512 chars).

        Args:
            chunks: List of LegalChunk objects
            batch_size: Number of chunks to process at once

        Returns:
            Number of chunks added
        """
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.text for chunk in batch]

            # Generate embeddings
            embeddings = await self.embed_batch_async(texts)

            # Build upsert vectors with text stored in metadata
            vectors = []
            for chunk_id, embedding, chunk in zip(ids, embeddings, batch):
                meta = _sanitize_metadata(chunk.metadata)
                meta["text"] = chunk.text
                vectors.append({"id": chunk_id, "values": embedding, "metadata": meta})

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=self.namespace)

            print(f"  Added {i + len(batch)}/{total_chunks} chunks")

        return total_chunks

    async def search(
        self, query: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Returns results in ChromaDB-compatible format so RetrievalService needs minimal changes:
        {"ids": [[...]], "distances": [[...]], "metadatas": [[...]], "documents": [[...]]}
        """
        query_embedding = await self.embed_text_async(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_metadata,
        )

        # Convert to ChromaDB-compatible format
        ids = []
        distances = []
        metadatas = []
        documents = []

        for match in results.get("matches", []):
            ids.append(match["id"])
            # Convert Pinecone similarity score to distance: distance = 1 - score
            distances.append(1 - match["score"])
            meta = dict(match.get("metadata", {}))
            text = meta.pop("text", "")
            metadatas.append(meta)
            documents.append(text)

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
            "documents": [documents],
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        stats = self.index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self.namespace, {})
        count = ns_stats.get("vector_count", 0)

        return {
            "name": self.index_name,
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "has_data": count > 0,
            "sample_metadata_keys": [],
        }

    def delete_collection(self):
        """Delete all vectors in the namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)

    def rebuild_collection(self):
        """Delete all vectors and start fresh (namespace is reusable)."""
        self.delete_collection()

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs from Pinecone.

        Args:
            ids: List of document IDs

        Returns:
            List of document dictionaries
        """
        result = self.index.fetch(ids=ids, namespace=self.namespace)

        documents = []
        for doc_id, vector_data in result.get("vectors", {}).items():
            meta = dict(vector_data.get("metadata", {}))
            text = meta.pop("text", "")
            documents.append({"id": doc_id, "text": text, "metadata": meta})

        return documents
