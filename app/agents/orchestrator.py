"""
Agent Orchestrator.
Coordinates the multi-agent workflow for processing legal queries.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langsmith import traceable

from app.agents.base import AgentContext

logger = logging.getLogger("mmara.orchestrator")
from app.agents.intake import IntakeAgent
from app.agents.legal import LegalAgent
from app.agents.safety import EmergencyHandler, ResponseValidator, SafetyAgent
from app.config import LEGAL_SYSTEM_PROMPT
from app.services.openai_client import OpenAIClient
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
        openai_client: OpenAIClient,
        redis_service: RedisService,
    ):
        self.retrieval_service = retrieval_service
        self.openai_client = openai_client
        self.redis_service = redis_service

        # Initialize agents
        self.intake_agent = IntakeAgent(openai_client=openai_client)
        self.safety_agent = SafetyAgent()
        self.legal_agent = LegalAgent(retrieval_service=retrieval_service, openai_client=openai_client)
        self.emergency_handler = EmergencyHandler(openai_client=openai_client)
        self.response_validator = ResponseValidator()

        # Track execution times
        self._execution_times: Dict[str, List[float]] = {}

    @traceable(name="orchestrator.process_query")
    async def process_query(
        self, query: str, session_id: Optional[str] = None, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent pipeline.
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())

        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info("=" * 60)
        logger.info("PIPELINE START | query='%s' | session=%s", query[:100], session_id[:8])
        logger.info("=" * 60)

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
            step_time = time.time()
            await self.intake_agent.execute(context)
            self._track_time("intake", start_time)
            logger.info(
                "STEP 1 INTAKE (%.3fs) | category=%s intent=%s urgency=%s is_emergency=%s",
                time.time() - step_time,
                context.metadata.get("category"),
                context.metadata.get("intent"),
                context.metadata.get("urgency"),
                context.metadata.get("is_emergency"),
            )

            # Step 2: Safety check
            step_time = time.time()
            safety_result = await self.safety_agent.execute(context)
            logger.info(
                "STEP 2 SAFETY (%.3fs) | safe=%s disclaimer=%s",
                time.time() - step_time,
                not safety_result.is_rejected(),
                bool(context.metadata.get("disclaimer")),
            )
            if safety_result.is_rejected():
                logger.warning("PIPELINE REJECTED by SafetyAgent: %s", safety_result.message)
                return self._format_response(
                    context=context,
                    response=safety_result.message or "Query not allowed.",
                    message_id=message_id,
                    session_id=session_id,
                    execution_time=time.time() - start_time,
                    rejected=True,
                )

            # Step 3: Legal retrieval
            step_time = time.time()
            await self.legal_agent.execute(context)
            self._track_time("legal", time.time())
            docs = context.metadata.get("retrieved_documents", [])
            logger.info(
                "STEP 3 LEGAL (%.3fs) | document_count=%d has_relevant_info=%s citations=%d",
                time.time() - step_time,
                len(docs),
                context.metadata.get("has_relevant_info"),
                len(context.metadata.get("citations", [])),
            )

            # Step 4: Handle emergency if needed
            if context.metadata.get("is_emergency"):
                step_time = time.time()
                await self.emergency_handler.execute(context)
                logger.info(
                    "STEP 4 EMERGENCY (%.3fs) | emergency response generated",
                    time.time() - step_time,
                )
            else:
                docs = context.metadata.get("retrieved_documents", [])
                # If no documents were retrieved, return honest fallback instead of hallucinating
                if not docs:
                    context.metadata["generated_response"] = (
                        "I wasn't able to find relevant information in the loaded legal documents "
                        "to answer your question accurately.\n\n"
                        "This may be because:\n"
                        "- The specific topic isn't covered in the indexed documents yet\n"
                        "- Try rephrasing your question with specific legal terms\n\n"
                        "---\n\n"
                        "**Disclaimer:** I am an AI assistant, not a qualified lawyer. "
                        "For serious legal matters, please consult a qualified lawyer."
                    )
                    logger.info("STEP 5 GENERATE (0.000s) | no documents found — returning fallback")
                else:
                    # Step 5: Generate response from retrieved documents
                    step_time = time.time()
                    await self._generate_response(context)
                    response_text = context.metadata.get("generated_response", "")
                    logger.info(
                        "STEP 5 GENERATE (%.3fs) | response_length=%d chars",
                        time.time() - step_time,
                        len(response_text),
                    )

            # Step 6: Validate response
            step_time = time.time()
            await self.response_validator.execute(context)
            logger.info(
                "STEP 6 VALIDATE (%.3fs) | validated=%s",
                time.time() - step_time,
                bool(context.metadata.get("generated_response")),
            )

            # Get final response
            response = context.metadata.get("generated_response", "")

            # Save to conversation history
            await self._save_to_history(session_id, query, response, message_id=message_id)

            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(
                "PIPELINE COMPLETE | total_time=%.3fs | response_length=%d",
                total_time,
                len(response),
            )
            logger.info("=" * 60)

            # Format response
            return self._format_response(
                context=context,
                response=response,
                message_id=message_id,
                session_id=session_id,
                execution_time=total_time,
            )

        except Exception as e:
            logger.error(
                "PIPELINE ERROR after %.3fs | error=%s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            return self._format_response(
                context=context,
                response="I apologize, but I encountered an error processing your request. Please try again.",
                message_id=message_id,
                session_id=session_id,
                execution_time=time.time() - start_time,
                error=str(e),
            )

    @traceable(name="orchestrator.generate_response")
    async def _generate_response(self, context: AgentContext):
        """Generate response using OpenAI LLM."""
        query = context.query
        documents = context.metadata.get("retrieved_documents", [])
        conversation_history = context.conversation_history or []

        response = await self.openai_client.generate_legal_response(
            query=query, context=documents, conversation_history=conversation_history
        )

        context.metadata["generated_response"] = response

    async def _save_to_history(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        message_id: str | None = None,
    ):
        """Save conversation to history."""
        user_msg_id = f"user_{message_id}" if message_id else str(uuid.uuid4())
        await self.redis_service.add_message_to_session(
            session_id,
            {
                "role": "user",
                "content": user_message,
                "message_id": user_msg_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        await self.redis_service.add_message_to_session(
            session_id,
            {
                "role": "assistant",
                "content": assistant_message,
                "message_id": message_id or str(uuid.uuid4()),
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
            "intent": context.metadata.get("intent", "general"),
            "document_count": context.metadata.get("document_count", 0),
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

        async for chunk in self.openai_client.stream_chat(messages):
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
