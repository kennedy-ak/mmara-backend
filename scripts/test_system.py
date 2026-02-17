"""
System Test Script
Test the MMara backend system components.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.orchestrator import AgentOrchestrator
from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.groq_client import GroqClient
from app.services.redis_client import RedisService
from app.services.retrieval import RetrievalService


async def test_embeddings():
    """Test embedding service."""
    print("\n=== Testing Embedding Service ===")

    service = EmbeddingService(persist_directory=settings.chroma_persist_directory_resolved)

    stats = service.get_collection_stats()
    print(f"Collection: {stats['name']}")
    print(f"Total documents: {stats['total_documents']}")

    if stats["total_documents"] == 0:
        print("Warning: No documents in collection. Run process_docs.py first.")
        return False

    # Test search
    print("\nTesting search...")
    results = await service.search("dangerous driving", n_results=3)
    print(f"Found {len(results['ids'][0])} results")

    return True


async def test_retrieval():
    """Test retrieval service."""
    print("\n=== Testing Retrieval Service ===")

    embedding_service = EmbeddingService(
        persist_directory=settings.chroma_persist_directory_resolved
    )

    retrieval_service = RetrievalService(embedding_service=embedding_service)

    # Test hybrid retrieval
    print("\nTesting hybrid retrieval...")
    results = await retrieval_service.retrieve(
        query="What is the penalty for dangerous driving?", n_results=5
    )

    print(f"Retrieved {len(results)} results")
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        source = metadata.get("source_file", "Unknown")
        score = result.get("rrf_score", 0)
        print(f"  {i}. {source} (score: {score:.4f})")
        print(f"     {result['text'][:100]}...")

    return True


async def test_groq():
    """Test Groq client."""
    print("\n=== Testing Groq Client ===")

    if not settings.groq_api_key:
        print("Groq API key not configured. Skipping Groq tests.")
        return False

    client = GroqClient(api_key=settings.groq_api_key)

    # Test classification
    print("\nTesting query classification...")
    classification = await client.classify_query("What are my rights during a police arrest?")
    print(f"Classification: {classification}")

    return True


async def test_orchestrator():
    """Test agent orchestrator."""
    print("\n=== Testing Agent Orchestrator ===")

    # Initialize services
    embedding_service = EmbeddingService(
        persist_directory=settings.chroma_persist_directory_resolved
    )
    retrieval_service = RetrievalService(embedding_service=embedding_service)

    groq_client = None
    if settings.groq_api_key:
        groq_client = GroqClient(api_key=settings.groq_api_key)

    redis_service = RedisService(url=settings.redis_url)
    await redis_service.connect()

    orchestrator = AgentOrchestrator(
        retrieval_service=retrieval_service, groq_client=groq_client, redis_service=redis_service
    )

    # Test query processing
    print("\nProcessing query: 'What is dangerous driving?'")
    result = await orchestrator.process_query(
        query="What is dangerous driving?", session_id="test-session"
    )

    print("\nResponse:")
    print(result["response"][:500] + "...")
    print(f"\nConfidence: {result['confidence']}")
    print(f"Category: {result['category']}")
    print(f"Citations: {len(result['citations'])}")

    await redis_service.disconnect()
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("MMara Backend System Tests")
    print("=" * 60)

    tests = [
        ("Embeddings", test_embeddings),
        ("Retrieval", test_retrieval),
        ("Groq", test_groq),
        ("Orchestrator", test_orchestrator),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for v in results.values() if v)
    print(f"\nPassed: {passed_count}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
