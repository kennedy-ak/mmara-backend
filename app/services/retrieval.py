"""
Hybrid Retrieval Service.
Combines semantic search and BM25 for improved retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.services.embeddings import EmbeddingService


class RetrievalService:
    """
    Hybrid retriever combining vector search and BM25.

    Uses reciprocal rank fusion (RRF) to combine results.
    """

    def __init__(self, embedding_service: EmbeddingService, alpha: float = 0.7, rrf_k: int = 60):
        """
        Args:
            embedding_service: Service for vector embeddings
            alpha: Weight for semantic search (0-1). BM25 weight = 1 - alpha
            rrf_k: Constant for reciprocal rank fusion
        """
        self.embedding_service = embedding_service
        self.alpha = alpha
        self.rrf_k = rrf_k

        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadatas: List[Dict] = []
        self._bm25_built = False

    async def build_bm25_index(self):
        """Build BM25 index from existing vector database."""
        if self._bm25_built:
            return

        print("Building BM25 index...")

        # Get all documents from ChromaDB
        collection = self.embedding_service.collection
        count = collection.count()

        # Fetch in batches
        batch_size = 1000
        all_docs = []
        all_ids = []
        all_metadata = []

        for offset in range(0, count, batch_size):
            batch = collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            all_docs.extend(batch["documents"])
            all_ids.extend(batch["ids"])
            all_metadata.extend(batch["metadatas"])

        # Tokenize documents for BM25
        loop = asyncio.get_event_loop()
        tokenized_docs = await loop.run_in_executor(
            None, lambda: [self._tokenize(doc) for doc in all_docs]
        )

        self.documents = all_docs
        self.doc_ids = all_ids
        self.metadatas = all_metadata
        self.bm25 = BM25Okapi(tokenized_docs)
        self._bm25_built = True

        print(f"✓ BM25 index built with {len(all_docs)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    async def _semantic_search(
        self, query: str, n_results: int, filter_metadata: Optional[Dict] = None
    ) -> List[tuple[str, float, Dict]]:
        """
        Perform semantic vector search.

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        results = await self.embedding_service.search(
            query=query, n_results=n_results, filter_metadata=filter_metadata
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]
                # Convert distance to similarity score
                similarity = 1 - distance
                formatted.append((doc_id, similarity, metadata))

        return formatted

    async def _keyword_search(self, query: str, n_results: int) -> List[tuple[str, float, Dict]]:
        """
        Perform BM25 keyword search.

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if not self._bm25_built:
            await self.build_bm25_index()

        # Tokenize query
        loop = asyncio.get_event_loop()
        tokenized_query = await loop.run_in_executor(None, lambda: self._tokenize(query))

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top indices
        top_indices = np.argsort(scores)[::-1][:n_results]

        # Format results
        formatted = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            metadata = self.metadatas[idx]
            formatted.append((doc_id, score, metadata))

        return formatted

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[tuple[str, float, Dict]],
        keyword_results: List[tuple[str, float, Dict]],
        n_results: int = 10,
    ) -> List[tuple[str, float, Dict]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1 / (k + rank)
        """
        # Score dictionaries
        scores = {}

        # Add semantic scores
        for rank, (doc_id, score, metadata) in enumerate(semantic_results, 1):
            rrf_score = 1 / (self.rrf_k + rank)
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf_score": 0,
                    "metadata": metadata,
                    "semantic_score": score,
                    "keyword_score": 0,
                }
            scores[doc_id]["rrf_score"] += self.alpha * rrf_score

        # Add keyword scores
        for rank, (doc_id, score, metadata) in enumerate(keyword_results, 1):
            rrf_score = 1 / (self.rrf_k + rank)
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf_score": 0,
                    "metadata": metadata,
                    "semantic_score": 0,
                    "keyword_score": score,
                }
            scores[doc_id]["rrf_score"] += (1 - self.alpha) * rrf_score
            scores[doc_id]["keyword_score"] = score

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[
            :n_results
        ]

        # Format output
        formatted = [
            (doc_id, data["rrf_score"], data["metadata"]) for doc_id, data in sorted_results
        ]

        return formatted

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval combining semantic and keyword search.

        Args:
            query: User query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter for semantic search
            use_hybrid: Whether to use hybrid search (vs semantic only)

        Returns:
            List of result dictionaries with text, metadata, and scores
        """
        if use_hybrid:
            # Get more results from each method for better fusion
            fetch_k = n_results * 3

            # Semantic search
            semantic_results = await self._semantic_search(
                query, n_results=fetch_k, filter_metadata=filter_metadata
            )

            # Keyword search
            keyword_results = await self._keyword_search(query, n_results=fetch_k)

            # Combine using RRF
            fused_results = self._reciprocal_rank_fusion(
                semantic_results, keyword_results, n_results=n_results
            )
        else:
            # Semantic only
            fused_results = await self._semantic_search(
                query, n_results=n_results, filter_metadata=filter_metadata
            )

        # Fetch full documents
        doc_ids = [r[0] for r in fused_results]
        fetched = self.embedding_service.collection.get(
            ids=doc_ids, include=["documents", "metadatas"]
        )

        # Build final results
        final_results = []
        id_to_index = {id_: i for i, id_ in enumerate(fetched["ids"])}

        for doc_id, rrf_score, _ in fused_results:
            idx = id_to_index.get(doc_id)
            if idx is not None:
                final_results.append(
                    {
                        "text": fetched["documents"][idx],
                        "metadata": fetched["metadatas"][idx],
                        "rrf_score": rrf_score,
                        "chunk_id": doc_id,
                    }
                )

        return final_results

    async def retrieve_by_category(
        self, query: str, category: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from a specific category."""
        return await self.retrieve(
            query=query, n_results=n_results, filter_metadata={"category": category}
        )

    async def rerank(
        self, query: str, results: List[Dict[str, Any]], groq_client, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using Groq LLM.

        Args:
            query: Original query
            results: Retrieved results
            groq_client: Groq client for reranking
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Use Groq to rerank
        reranked = await groq_client.rerank_results(query, results, top_k)

        return reranked

    async def retrieve_with_rerank(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        groq_client=None,
        fetch_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with LLM reranking.

        Args:
            query: User query
            n_results: Final number of results to return
            filter_metadata: Optional metadata filter
            groq_client: Groq client for reranking
            fetch_k: Number of results to fetch before reranking

        Returns:
            Final reranked results
        """
        # Fetch more results
        results = await self.retrieve(
            query=query, n_results=fetch_k, filter_metadata=filter_metadata
        )

        # Rerank if Groq client available
        if groq_client:
            results = await self.rerank(query, results, groq_client, n_results)
        else:
            results = results[:n_results]

        return results
