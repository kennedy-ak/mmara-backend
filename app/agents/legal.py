"""
Legal Knowledge Agent.
Handles legal query formulation, retrieval, and citation extraction.
"""

from typing import Any, Dict, List, Optional

from app.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from app.services.retrieval import RetrievalService


class LegalAgent(BaseAgent):
    """
    Legal Knowledge Agent.

    Responsibilities:
    - Formulates legal query for retrieval
    - Interprets retrieved documents
    - Extracts relevant sections
    - Prepares legal citations
    """

    def __init__(self, retrieval_service: RetrievalService, groq_client=None):
        super().__init__()
        self.retrieval_service = retrieval_service
        self.groq_client = groq_client

    @property
    def name(self) -> str:
        return "legal"

    @property
    def description(self) -> str:
        return "Handles legal knowledge retrieval and interpretation"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Process legal query through retrieval.

        Args:
            context: Agent context

        Returns:
            AgentResult with retrieved documents and citations
        """
        query = context.query
        category = context.metadata.get("category", "general")

        # Formulate optimized retrieval query
        retrieval_query = self._formulate_query(query, context)

        # Retrieve documents
        results = await self._retrieve(
            retrieval_query, category, context.metadata.get("urgency", "low")
        )

        if not results:
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data={"documents": [], "citations": [], "has_relevant_info": False},
                message="No relevant legal documents found",
            )

        # Extract citations
        citations = self._extract_citations(results)

        # Store in context
        context.metadata["retrieved_documents"] = results
        context.metadata["citations"] = citations
        context.metadata["has_relevant_info"] = len(results) > 0

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={
                "documents": results,
                "citations": citations,
                "has_relevant_info": True,
                "document_count": len(results),
            },
            confidence=min(0.5 + (len(results) * 0.1), 1.0),
        )

    def _formulate_query(self, query: str, context: AgentContext) -> str:
        """
        Formulate an optimized query for retrieval.

        Args:
            query: Original user query
            context: Agent context with metadata

        Returns:
            Optimized query string
        """
        # Add category-specific terms
        urgency = context.metadata.get("urgency", "low")

        # For urgent queries, focus on procedural/constitutional rights
        if urgency in ["high", "critical"]:
            urgency_terms = "rights procedure arrest detention bail"
            return f"{query} {urgency_terms}"

        # For regular queries, use as-is
        return query

    async def _retrieve(self, query: str, category: str, urgency: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant legal documents.

        Args:
            query: Optimized query
            category: Legal category
            urgency: Urgency level

        Returns:
            List of retrieved documents
        """
        # Determine retrieval parameters
        top_k = 10 if urgency in ["high", "critical"] else 5

        # Use category filter if specified
        filter_metadata = {"category": category} if category != "general" else None

        # For urgent queries, skip reranking for speed
        use_rerank = urgency not in ["high", "critical"]

        # Retrieve
        if use_rerank and self.groq_client:
            results = await self.retrieval_service.retrieve_with_rerank(
                query=query,
                n_results=top_k,
                filter_metadata=filter_metadata,
                groq_client=self.groq_client,
            )
        else:
            results = await self.retrieval_service.retrieve(
                query=query, n_results=top_k, filter_metadata=filter_metadata
            )

        return results

    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract legal citations from retrieved documents.

        Args:
            results: Retrieved documents

        Returns:
            List of citation dictionaries
        """
        citations = []

        for result in results:
            metadata = result.get("metadata", {})

            citation = {
                "act": "",
                "section": "",
                "text": result.get("text", "")[:200],
                "source_file": metadata.get("source_file", ""),
            }

            # Build citation string
            if act_number := metadata.get("act_number"):
                citation["act"] = f"Act {act_number}"

            if section := metadata.get("section_header", ""):
                citation["section"] = section

            if legislative_instrument := metadata.get("legislative_instrument"):
                citation["act"] = legislative_instrument

            if year := metadata.get("year"):
                if citation["act"]:
                    citation["act"] += f" ({year})"
                else:
                    citation["act"] = year

            citations.append(citation)

        return citations


class CitationFormatter(BaseAgent):
    """
    Format legal citations for display in responses.
    """

    @property
    def name(self) -> str:
        return "citation_formatter"

    @property
    def description(self) -> str:
        return "Formats legal citations for user-friendly display"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Format citations from retrieved documents.

        Args:
            context: Agent context

        Returns:
            AgentResult with formatted citations
        """
        documents = context.metadata.get("retrieved_documents", [])
        citations = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            formatted = self._format_citation(metadata, doc.get("text", ""))
            if formatted:
                citations.append(formatted)

        context.metadata["formatted_citations"] = citations

        return AgentResult(
            status=AgentStatus.SUCCESS, data={"citations": citations}, confidence=1.0
        )

    def _format_citation(self, metadata: Dict, text: str) -> Optional[Dict[str, str]]:
        """Format a single citation."""
        parts = []

        # Source file
        if source := metadata.get("source_file"):
            parts.append(source)

        # Act number
        if act := metadata.get("act_number"):
            parts.append(f"Act {act}")

        # Year
        if year := metadata.get("year"):
            parts.append(f"({year})")

        # Section
        if section := metadata.get("section_header"):
            parts.append(section)

        if not parts:
            return None

        return {"citation": " - ".join(parts), "text": text[:300], "metadata": metadata}
