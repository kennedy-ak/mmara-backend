"""
Agent Orchestrator.
Coordinates the multi-agent workflow for processing legal queries.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.base import AgentContext
from app.agents.intake import IntakeAgent
from app.agents.legal import LegalAgent
from app.agents.safety import EmergencyHandler, ResponseValidator, SafetyAgent
from app.config import LEGAL_SYSTEM_PROMPT
from app.services.groq_client import GroqClient
from app.services.redis_client import RedisService
from app.services.retrieval import RetrievalService


class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow for legal queries.

    Workflow:
    1. IntakeAgent → classify query
    2. SafetyAgent → check constraints
    3. LegalAgent → retrieve + interpret
    4. EmergencyHandler → if emergency
    5. GroqLLMAgent → generate response
    6. ResponseValidator → validate response
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        groq_client: GroqClient,
        redis_service: RedisService,
    ):
        self.retrieval_service = retrieval_service
        self.groq_client = groq_client
        self.redis_service = redis_service

        # Initialize agents
        self.intake_agent = IntakeAgent(groq_client=groq_client)
        self.safety_agent = SafetyAgent()
        self.legal_agent = LegalAgent(retrieval_service=retrieval_service, groq_client=groq_client)
        self.emergency_handler = EmergencyHandler(groq_client=groq_client)
        self.response_validator = ResponseValidator()

        # Track execution times
        self._execution_times: Dict[str, List[float]] = {}

    async def process_query(
        self, query: str, session_id: Optional[str] = None, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent pipeline.

        Args:
            query: User query
            session_id: Optional session ID for conversation history
            user_id: Optional user ID

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Get conversation history
        conversation_history = await self.redis_service.get_session(session_id)

        # Create context
        context = AgentContext(
            query=query,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
        )

        try:
            # Step 1: Intake - Classify query
            await self.intake_agent.execute(context)
            self._track_time("intake", start_time)

            # Step 2: Safety check
            safety_result = await self.safety_agent.execute(context)
            if safety_result.is_rejected():
                return self._format_response(
                    context=context,
                    response=safety_result.message or "Query not allowed.",
                    message_id=message_id,
                    session_id=session_id,
                    execution_time=time.time() - start_time,
                    rejected=True,
                )

            # Step 3: Legal retrieval
            await self.legal_agent.execute(context)
            self._track_time("legal", time.time())

            # Step 4: Handle emergency if needed
            if context.metadata.get("is_emergency"):
                await self.emergency_handler.execute(context)
            else:
                # Step 5: Generate response
                await self._generate_response(context)

            # Step 6: Validate response
            await self.response_validator.execute(context)

            # Get final response
            response = context.metadata.get("generated_response", "")

            # Save to conversation history
            await self._save_to_history(session_id, query, response)

            # Format response
            return self._format_response(
                context=context,
                response=response,
                message_id=message_id,
                session_id=session_id,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            # Log error and return error response
            return self._format_response(
                context=context,
                response="I apologize, but I encountered an error processing your request. Please try again.",
                message_id=message_id,
                session_id=session_id,
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _generate_response(self, context: AgentContext):
        """Generate response using Groq LLM."""
        query = context.query
        documents = context.metadata.get("retrieved_documents", [])
        conversation_history = context.conversation_history or []

        # Generate legal response
        response = await self.groq_client.generate_legal_response(
            query=query, context=documents, conversation_history=conversation_history
        )

        context.metadata["generated_response"] = response

    async def _save_to_history(self, session_id: str, user_message: str, assistant_message: str):
        """Save conversation to history."""
        await self.redis_service.add_message_to_session(
            session_id,
            {"role": "user", "content": user_message, "timestamp": datetime.utcnow().isoformat()},
        )
        await self.redis_service.add_message_to_session(
            session_id,
            {
                "role": "assistant",
                "content": assistant_message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _format_response(
        self,
        context: AgentContext,
        response: str,
        message_id: str,
        session_id: str,
        execution_time: float,
        rejected: bool = False,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format the final response."""
        return {
            "response": response,
            "session_id": session_id,
            "message_id": message_id,
            "citations": context.metadata.get("citations", []),
            "confidence": self._calculate_confidence(context),
            "category": context.metadata.get("category", "general"),
            "urgency": context.metadata.get("urgency", "low"),
            "is_emergency": context.metadata.get("is_emergency", False),
            "disclaimer": context.metadata.get("disclaimer", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": execution_time * 1000,
            "rejected": rejected,
            "error": error,
        }

    def _calculate_confidence(self, context: AgentContext) -> float:
        """Calculate overall confidence score."""
        has_documents = context.metadata.get("has_relevant_info", False)
        document_count = context.metadata.get("document_count", 0)
        is_emergency = context.metadata.get("is_emergency", False)

        if is_emergency:
            return 0.7  # Lower confidence for emergencies

        if not has_documents:
            return 0.3

        # Base confidence from document count
        confidence = min(0.5 + (document_count * 0.1), 0.95)

        return confidence

    def _track_time(self, stage: str, start_time: float):
        """Track execution time for a stage."""
        elapsed = time.time() - start_time
        if stage not in self._execution_times:
            self._execution_times[stage] = []
        self._execution_times[stage].append(elapsed)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        for stage, times in self._execution_times.items():
            if times:
                stats[stage] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times),
                }
        return stats

    async def stream_query(
        self, query: str, session_id: Optional[str] = None, user_id: Optional[int] = None
    ):
        """
        Stream a query response (for WebSocket).

        Args:
            query: User query
            session_id: Optional session ID
            user_id: Optional user ID

        Yields:
            Response chunks
        """
        # For now, just process normally and return
        result = await self.process_query(query, session_id, user_id)
        yield result


class StreamingOrchestrator(AgentOrchestrator):
    """
    Orchestrator with streaming support for real-time responses.
    """

    async def stream_query_with_chunks(
        self, query: str, session_id: Optional[str] = None, user_id: Optional[int] = None
    ):
        """
        Stream query response in chunks.

        Yields response chunks as they're generated.
        """
        message_id = str(uuid.uuid4())
        start_time = time.time()

        if not session_id:
            session_id = str(uuid.uuid4())

        # Get conversation history
        conversation_history = await self.redis_service.get_session(session_id)

        # Create context
        context = AgentContext(
            query=query,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
        )

        # Step 1: Intake
        yield self._chunk("status", {"stage": "classifying", "message": "Analyzing query..."})
        intake_result = await self.intake_agent.execute(context)
        yield self._chunk("classification", intake_result.data)

        # Step 2: Safety
        safety_result = await self.safety_agent.execute(context)
        if safety_result.is_rejected():
            yield self._chunk("error", {"message": safety_result.message})
            return

        # Step 3: Retrieval
        yield self._chunk(
            "status", {"stage": "retrieving", "message": "Searching legal documents..."}
        )
        legal_result = await self.legal_agent.execute(context)
        yield self._chunk("retrieval", {"count": len(legal_result.data.get("documents", []))})

        # Step 4: Generate response (streaming)
        yield self._chunk("status", {"stage": "generating", "message": "Generating response..."})

        documents = context.metadata.get("retrieved_documents", [])

        # Stream from Groq
        full_response = ""
        context_str = self._format_context_for_llm(documents)

        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ]

        async for chunk in self.groq_client.stream_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield self._chunk("token", {"content": content})

        # Save and validate
        context.metadata["generated_response"] = full_response
        await self.response_validator.execute(context)

        # Final chunk
        yield self._chunk(
            "complete",
            {
                "response": context.metadata.get("generated_response"),
                "citations": context.metadata.get("citations", []),
                "session_id": session_id,
                "message_id": message_id,
                "response_time_ms": (time.time() - start_time) * 1000,
            },
        )

        # Save to history
        await self._save_to_history(session_id, query, full_response)

    def _chunk(self, chunk_type: str, data: Dict) -> str:
        """Format a chunk for streaming."""
        return json.dumps({"type": chunk_type, "data": data})

    def _format_context_for_llm(self, documents: List[Dict]) -> str:
        """Format documents for LLM context."""
        if not documents:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            text = doc.get("text", "")[:500]
            formatted.append(f"[{i}] {source}: {text}")

        return "\n\n".join(formatted)
