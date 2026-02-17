"""
Safety Guard Agent.
Validates queries and responses for safety and policy compliance.
"""

from typing import Dict, List

from app.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent


class SafetyAgent(BaseAgent):
    """
    Safety Guard Agent.

    Responsibilities:
    - Checks for emergency indicators
    - Enforces disclaimer requirements
    - Refuses inappropriate requests
    - Validates legal boundaries
    """

    # Restricted topics
    RESTRICTED_TOPICS = [
        "violence",
        "harm",
        "illegal activity",
        "evading police",
        "resisting arrest",
        "bribery",
        "corruption",
    ]

    # Emergency indicators
    EMERGENCY_INDICATORS = [
        "arresting me now",
        "being arrested",
        "police right now",
        "detained right now",
        "emergency",
        "help immediately",
    ]

    DISCLAIMER = (
        "I am an AI assistant, not a qualified lawyer. "
        "This information is for educational purposes only "
        "and does not constitute legal advice. For serious "
        "legal matters, please consult a qualified lawyer."
    )

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "safety"

    @property
    def description(self) -> str:
        return "Validates queries and responses for safety"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Validate query for safety.

        Args:
            context: Agent context

        Returns:
            AgentResult with validation status
        """
        query = context.query.lower()

        # Check for restricted topics
        if self._has_restricted_content(query):
            return AgentResult(
                status=AgentStatus.REJECTED,
                data={"reason": "restricted_content"},
                message="I cannot assist with requests that may involve illegal activities.",
                next_agent=None,
            )

        # Check for emergency
        is_emergency = self._is_emergency(query)
        context.metadata["is_emergency"] = is_emergency

        # Store disclaimer
        context.metadata["disclaimer"] = self.DISCLAIMER

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={"is_emergency": is_emergency, "disclaimer": self.DISCLAIMER, "query_safe": True},
        )

    def _has_restricted_content(self, query: str) -> bool:
        """Check if query contains restricted content."""
        restricted_phrases = [
            "how to evade",
            "how to avoid",
            "how to escape",
            "bribe",
            "pay off",
            "corruption",
            "hurt someone",
            "violence against",
        ]

        query_lower = query.lower()
        return any(phrase in query_lower for phrase in restricted_phrases)

    def _is_emergency(self, query: str) -> bool:
        """Check if query indicates an emergency."""
        return any(indicator in query.lower() for indicator in self.EMERGENCY_INDICATORS)


class ResponseValidator(BaseAgent):
    """
    Validates generated responses for safety and quality.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "response_validator"

    @property
    def description(self) -> str:
        return "Validates generated responses"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Validate the generated response.

        Args:
            context: Agent context

        Returns:
            AgentResult with validation status
        """
        response = context.metadata.get("generated_response", "")

        # Check for required elements
        has_disclaimer = self._has_disclaimer(response)
        is_emergency = context.metadata.get("is_emergency", False)

        # For emergencies, ensure practical guidance
        if is_emergency:
            has_practical_guidance = self._has_practical_guidance(response)
            if not has_practical_guidance:
                context.metadata["needs_emergency_formatting"] = True

        # Ensure disclaimer is present
        if not has_disclaimer:
            response = self._add_disclaimer(response)
            context.metadata["generated_response"] = response

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={"response_validated": True, "disclaimer_added": not has_disclaimer},
        )

    def _has_disclaimer(self, response: str) -> bool:
        """Check if response contains disclaimer."""
        disclaimer_indicators = [
            "not a lawyer",
            "not a qualified lawyer",
            "does not constitute legal advice",
            "educational purposes",
        ]
        return any(indicator in response.lower() for indicator in disclaimer_indicators)

    def _has_practical_guidance(self, response: str) -> bool:
        """Check if response has practical immediate guidance."""
        guidance_indicators = [
            "right now",
            "immediately",
            "should",
            "do the following",
            "next steps",
            "contact",
            "remain calm",
        ]
        return any(indicator in response.lower() for indicator in guidance_indicators)

    def _add_disclaimer(self, response: str) -> str:
        """Add disclaimer to response."""
        disclaimer = (
            "\n\n---\n\n"
            "**Disclaimer:** I am an AI assistant, not a qualified lawyer. "
            "This information is for educational purposes only and does not "
            "constitute legal advice. For serious legal matters, please "
            "consult a qualified lawyer."
        )
        return response + disclaimer


class EmergencyHandler(BaseAgent):
    """
    Handles emergency queries with prioritized responses.
    """

    def __init__(self, groq_client=None):
        super().__init__()
        self.groq_client = groq_client

    @property
    def name(self) -> str:
        return "emergency_handler"

    @property
    def description(self) -> str:
        return "Handles emergency queries with immediate guidance"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Generate emergency response.

        Args:
            context: Agent context

        Returns:
            AgentResult with emergency response
        """
        query = context.query
        documents = context.metadata.get("retrieved_documents", [])

        # Generate emergency response
        if self.groq_client:
            response = await self.groq_client.generate_emergency_response(query, documents)
        else:
            response = self._generate_fallback_emergency_response(query, documents)

        context.metadata["generated_response"] = response
        context.metadata["is_emergency_response"] = True

        return AgentResult(
            status=AgentStatus.SUCCESS, data={"emergency_response": response, "urgency": "critical"}
        )

    def _generate_fallback_emergency_response(self, query: str, documents: List[Dict]) -> str:
        """Generate fallback emergency response without LLM."""
        return """This sounds like an urgent situation. Here's what you should know right now:

**Your Basic Rights:**
- You have the right to remain silent
- You have the right to know why you're being arrested
- You have the right to a lawyer
- Do not resist arrest, even if you believe it's unlawful

**Immediate Steps:**
1. Remain calm and polite
2. Clearly state: "I wish to remain silent"
3. Ask: "Am I free to go?"
4. Request a lawyer immediately
5. Do not sign anything without a lawyer

**Contact a lawyer as soon as possible.**

---
*I am an AI assistant, not a qualified lawyer. This is general information, not legal advice.*"""
