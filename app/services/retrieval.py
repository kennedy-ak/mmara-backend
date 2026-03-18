"""
Hybrid Retrieval Service.
Combines semantic search and BM25 for improved retrieval.
"""

import asyncio
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from langsmith import traceable
from rank_bm25 import BM25Okapi

from app.services.embeddings import EmbeddingService

logger = logging.getLogger("mmara.retrieval")

# Timeout for BM25 index building (seconds)
BM25_BUILD_TIMEOUT = 120

# Cache directory for BM25 index
BM25_CACHE_DIR = Path("/tmp/bm25_cache")
BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class RetrievalService:
    """
    Hybrid retriever combining vector search and BM25.

    Uses reciprocal rank fusion (RRF) to combine results.
    Falls back to semantic-only search if BM25 index is unavailable.
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
        self._bm25_building = False
        self._bm25_failed = False

        # Cache key based on index/namespace to detect changes
        self._cache_key = self._generate_cache_key()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for the current Pinecone index."""
        # Use index name and namespace as cache key
        index_name = getattr(self.embedding_service.index, 'name', 'default')
        namespace = self.embedding_service.namespace or 'default'
        key = f"{index_name}:{namespace}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> Path:
        """Get the cache file path for this index."""
        return BM25_CACHE_DIR / f"bm25_{self._cache_key}.pkl"

    async def build_bm25_index(self):
        """Build BM25 index from Pinecone vectors with timeout."""
        if self._bm25_built or self._bm25_building:
            return

        # Try loading from cache first
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                logger.info(f"Loading BM25 index from cache: {cache_path.name}")
                await asyncio.get_event_loop().run_in_executor(
                    None, self._load_from_cache, cache_path
                )
                if self._bm25_built:
                    logger.info(f"BM25 index loaded from cache with {len(self.documents)} documents")
                    return
            except Exception as e:
                logger.warning(f"Failed to load BM25 from cache: {e}. Building from scratch.")

        self._bm25_building = True
        logger.info("Building BM25 index from Pinecone...")

        try:
            index = self.embedding_service.index
            namespace = self.embedding_service.namespace

            # Run with timeout to prevent hanging
            all_ids, all_docs, all_metadata = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_all_documents, index, namespace
                ),
                timeout=BM25_BUILD_TIMEOUT,
            )

            if not all_docs:
                self._bm25_built = True
                logger.info("BM25 index: no documents found")
                return

            # Tokenize documents for BM25
            tokenized_docs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [self._tokenize(doc) for doc in all_docs]
            )

            self.documents = all_docs
            self.doc_ids = all_ids
            self.metadatas = all_metadata
            self.bm25 = BM25Okapi(tokenized_docs)
            self._bm25_built = True

            logger.info(f"BM25 index built with {len(all_docs)} documents")

            # Save to cache for next time
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_to_cache, cache_path
            )
            logger.info(f"BM25 index cached to {cache_path.name}")

        except asyncio.TimeoutError:
            logger.warning(
                f"BM25 index build timed out after {BM25_BUILD_TIMEOUT}s. "
                "Falling back to semantic-only search."
            )
            self._bm25_failed = True
        except Exception as e:
            logger.warning(f"BM25 index build failed: {e}. Falling back to semantic-only search.")
            self._bm25_failed = True
        finally:
            self._bm25_building = False

    def _save_to_cache(self, cache_path: Path):
        """Save BM25 index to disk."""
        cache_data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
            "cache_key": self._cache_key,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_from_cache(self, cache_path: Path):
        """Load BM25 index from disk."""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Verify cache key matches
        if cache_data.get("cache_key") != self._cache_key:
            logger.warning("BM25 cache key mismatch - ignoring cached index")
            return

        self.bm25 = cache_data["bm25"]
        self.documents = cache_data["documents"]
        self.doc_ids = cache_data["doc_ids"]
        self.metadatas = cache_data["metadatas"]
        self._bm25_built = True

    def _fetch_all_documents(self, index, namespace):
        """Synchronous helper to fetch all documents from Pinecone (runs in thread)."""
        all_docs = []
        all_ids = []
        all_metadata = []
        batch_size = 100
        id_batch = []
        total_fetched = 0

        for batch_num, id_list in enumerate(index.list(namespace=namespace), 1):
            for vec_id in id_list:
                id_batch.append(vec_id)
                if len(id_batch) >= batch_size:
                    fetched = index.fetch(ids=id_batch, namespace=namespace)
                    for vid, vdata in fetched.get("vectors", {}).items():
                        meta = dict(vdata.get("metadata", {}))
                        text = meta.pop("text", "")
                        all_ids.append(vid)
                        all_docs.append(text)
                        all_metadata.append(meta)
                        total_fetched += 1
                    id_batch = []
                    # Log progress every 500 docs
                    if total_fetched % 500 == 0:
                        logger.info(f"BM25 build: fetched {total_fetched} documents...")

        if id_batch:
            fetched = index.fetch(ids=id_batch, namespace=namespace)
            for vid, vdata in fetched.get("vectors", {}).items():
                meta = dict(vdata.get("metadata", {}))
                text = meta.pop("text", "")
                all_ids.append(vid)
                all_docs.append(text)
                all_metadata.append(meta)

        logger.info(f"BM25 build: finished fetching {total_fetched} documents")
        return all_ids, all_docs, all_metadata

    def invalidate_bm25_index(self):
        """Reset BM25 cache so it rebuilds on next search."""
        self._bm25_built = False
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self._bm25_failed = False

        # Also delete cache file
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Deleted BM25 cache: {cache_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete BM25 cache: {e}")

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
        if not self._bm25_built and not self._bm25_failed:
            await self.build_bm25_index()

        if not self.bm25:
            return []

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

    @traceable(name="retrieval.retrieve")
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

            # Keyword search (returns [] if BM25 failed/unavailable)
            keyword_results = await self._keyword_search(query, n_results=fetch_k)

            if keyword_results:
                # Combine using RRF
                fused_results = self._reciprocal_rank_fusion(
                    semantic_results, keyword_results, n_results=n_results
                )
            else:
                # BM25 unavailable — use semantic results only
                fused_results = semantic_results[:n_results]
        else:
            # Semantic only
            fused_results = await self._semantic_search(
                query, n_results=n_results, filter_metadata=filter_metadata
            )

        # Fetch full documents from Pinecone
        doc_ids = [r[0] for r in fused_results]
        if not doc_ids:
            return []

        loop = asyncio.get_event_loop()
        fetched_raw = await loop.run_in_executor(
            None,
            lambda: self.embedding_service.index.fetch(
                ids=doc_ids, namespace=self.embedding_service.namespace
            ),
        )
        vectors = fetched_raw.get("vectors", {})

        # Build final results
        final_results = []
        for doc_id, rrf_score, _ in fused_results:
            vdata = vectors.get(doc_id)
            if vdata:
                meta = dict(vdata.get("metadata", {}))
                text = meta.pop("text", "")
                final_results.append(
                    {
                        "text": text,
                        "metadata": meta,
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
        self, query: str, results: List[Dict[str, Any]], openai_client, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using OpenAI LLM.

        Args:
            query: Original query
            results: Retrieved results
            openai_client: OpenAI client for reranking
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results or not openai_client:
            return results[:top_k] if results else []

        reranked = await openai_client.rerank_results(query, results, top_k)

        return reranked

    async def retrieve_with_rerank(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        openai_client=None,
        fetch_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with LLM reranking.

        Args:
            query: User query
            n_results: Final number of results to return
            filter_metadata: Optional metadata filter
            openai_client: OpenAI client for reranking
            fetch_k: Number of results to fetch before reranking

        Returns:
            Final reranked results
        """
        results = await self.retrieve(
            query=query, n_results=fetch_k, filter_metadata=filter_metadata
        )

        if openai_client:
            results = await self.rerank(query, results, openai_client, n_results)
        else:
            results = results[:n_results]

        return results
