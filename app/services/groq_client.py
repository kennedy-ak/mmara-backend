"""
Groq API Client Service.
Handles LLM interactions using Groq's fast inference.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from groq import AsyncGroq
from groq.types.chat import ChatCompletion, ChatCompletionChunk

from app.config import LEGAL_SYSTEM_PROMPT, settings


class GroqClient:
    """
    Client for interacting with Groq API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("Groq API key is required")

        self.client = AsyncGroq(api_key=self.api_key)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatCompletion:
        """
        Send a chat completion request.

        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            ChatCompletion response
        """
        response = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            response_format=response_format,
        )

        return response

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream a chat completion response.

        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            ChatCompletionChunk objects
        """
        stream = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            yield chunk

    async def generate_legal_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a legal response based on query and retrieved context.

        Args:
            query: User query
            context: Retrieved document chunks
            conversation_history: Previous conversation messages

        Returns:
            Generated legal response
        """
        # Build context string
        context_str = self._format_context(context)

        # Build messages
        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ]

        # Add conversation history if available
        if conversation_history:
            messages = [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]
            # Add last few messages for context
            for msg in conversation_history[-6:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            # Add current query with context
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on the following legal context, please answer: {query}\n\nContext:\n{context_str}",
                }
            )

        response = await self.chat(messages)

        return response.choices[0].message.content or ""

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify the user query.

        Args:
            query: User query

        Returns:
            Dictionary with classification results
        """
        classification_prompt = """Classify the following legal query according to these categories:

1. **Intent**: question, emergency, clarification, general
2. **Category**: criminal, road_traffic, general
3. **Urgency**: low, medium, high, critical
4. **Is Emergency**: true or false

Return ONLY a valid JSON object like:
{"intent": "question", "category": "criminal", "urgency": "medium", "is_emergency": false, "reasoning": "brief explanation"}

Query: {query}"""

        response = await self.chat(
            messages=[{"role": "user", "content": classification_prompt.format(query=query)}],
            response_format={"type": "json_object"},
        )

        import json

        try:
            return json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {
                "intent": "general",
                "category": "general",
                "urgency": "low",
                "is_emergency": False,
                "reasoning": "Classification failed",
            }

    async def rerank_results(
        self, query: str, results: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using LLM.

        Args:
            query: Original query
            results: Retrieved results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        rerank_prompt = """You are a legal search reranker. Given a query and search results, rank them by relevance.

Query: {query}

Results:
{results}

Return ONLY a valid JSON array with the indices of the top {top_k} most relevant results in order, like:
[0, 3, 1, 4, 2]"""

        # Format results for the prompt
        results_str = ""
        for i, result in enumerate(results):
            text_preview = result.get("text", "")[:200]
            metadata = result.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            results_str += f"\n{i}. {source}: {text_preview}..."

        response = await self.chat(
            messages=[
                {
                    "role": "user",
                    "content": rerank_prompt.format(
                        query=query, results=results_str, top_k=min(top_k, len(results))
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )

        import json

        try:
            ranking = json.loads(response.choices[0].message.content or "[]")
            # Get ranking indices
            indices = ranking.get("ranking", list(range(len(results))))
            if not isinstance(indices, list):
                indices = list(range(len(results)))

            # Reorder results based on ranking
            reranked = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(results):
                    reranked.append(results[idx])

            return reranked
        except (json.JSONDecodeError, KeyError, TypeError):
            return results[:top_k]

    async def extract_citations(
        self, response: str, context: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract legal citations from the response.

        Args:
            response: Generated response text
            context: Retrieved documents

        Returns:
            List of citation dictionaries
        """
        citations_prompt = """Extract legal citations from this response. For each citation, provide:
- Act name/number
- Section (if available)
- The relevant text

Response: {response}

Available context:
{context}

Return ONLY a valid JSON array like:
[{{"act": "Act 29", "section": "1", "text": "relevant text"}}]"""

        context_str = self._format_context(context[:5])

        response_obj = await self.chat(
            messages=[
                {
                    "role": "user",
                    "content": citations_prompt.format(response=response, context=context_str),
                }
            ],
            response_format={"type": "json_object"},
        )

        import json

        try:
            citations = json.loads(response_obj.choices[0].message.content or "[]")
            return citations.get("citations", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved context into a string."""
        if not context:
            return "No relevant legal documents found."

        formatted = []
        for i, doc in enumerate(context, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            act = metadata.get("act_number", "")
            section = metadata.get("section_header", "")
            text = doc.get("text", "")[:500]

            citation = f"{source}"
            if act:
                citation += f" (Act {act})"
            if section:
                citation += f" - {section}"

            formatted.append(f"[{i}] {citation}\n{text}")

        return "\n\n".join(formatted)

    async def generate_emergency_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate an emergency response.

        Args:
            query: User query
            context: Retrieved documents

        Returns:
            Emergency response with immediate guidance
        """
        emergency_prompt = """This user may be in an emergency situation. Provide immediate, practical guidance.

User Query: {query}

Relevant Legal Context:
{context}

Provide:
1. Immediate actions the user should take
2. Their basic rights in this situation
3. What they should NOT do
4. When to contact a lawyer

Keep it clear, concise, and actionable."""

        context_str = self._format_context(context)

        response = await self.chat(
            messages=[
                {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": emergency_prompt.format(query=query, context=context_str),
                },
            ]
        )

        return response.choices[0].message.content or ""
