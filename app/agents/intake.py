"""
Intake Agent.
Analyzes user query to determine intent, category, and urgency.
"""

from typing import Any, Dict

from app.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from app.config import LegalCategory, QueryCategory, UrgencyLevel


class IntakeAgent(BaseAgent):
    """
    Intake/Classification Agent.

    Analyzes user query to determine:
    - Intent (question, emergency, clarification, general)
    - Category (traffic, criminal, general)
    - Urgency level (low, medium, high, critical)
    - Language (English, future: Twi)
    """

    def __init__(self, groq_client=None):
        super().__init__()
        self.groq_client = groq_client

    @property
    def name(self) -> str:
        return "intake"

    @property
    def description(self) -> str:
        return "Classifies user queries by intent, category, and urgency"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Process the query through classification.

        Args:
            context: Agent context

        Returns:
            AgentResult with classification data
        """
        query = context.query.lower()

        # Quick keyword-based classification for speed
        classification = self._quick_classify(query)

        # Override with LLM if available and needed
        if self.groq_client and classification.get("urgency") in ["high", "critical"]:
            llm_classification = await self.groq_client.classify_query(context.query)
            # Merge classifications (prefer LLM for complex cases)
            classification.update(llm_classification)

        # Store classification in context
        context.metadata["classification"] = classification
        context.metadata["intent"] = classification.get("intent", QueryCategory.GENERAL)
        context.metadata["category"] = classification.get("category", LegalCategory.GENERAL)
        context.metadata["urgency"] = classification.get("urgency", UrgencyLevel.LOW)
        context.metadata["is_emergency"] = classification.get("is_emergency", False)

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=classification,
            confidence=classification.get("confidence", 0.8),
        )

    def _quick_classify(self, query: str) -> Dict[str, Any]:
        """
        Quick keyword-based classification.

        Args:
            query: Lowercase query text

        Returns:
            Classification dictionary
        """
        # Emergency indicators
        emergency_keywords = [
            "arrest",
            "arresting",
            "detained",
            "detaining",
            "police",
            "search",
            "searching",
            "seize",
            "seizing",
            "confiscate",
            "rights",
            "right now",
            "emergency",
            "help me",
            "urgent",
            "accused",
            "charge",
            "charging",
        ]

        # Criminal law indicators
        criminal_keywords = [
            "crime",
            "criminal",
            "offence",
            "offense",
            "arrest",
            "bail",
            "detention",
            "search",
            "seizure",
            "evidence",
            "court",
            "judge",
            "lawyer",
            "attorney",
            "prosecution",
        ]

        # Road traffic indicators
        traffic_keywords = [
            "traffic",
            "road",
            "driver",
            "driving",
            "vehicle",
            "car",
            "license",
            "accident",
            "insurance",
            "speed",
            "drink",
            "dangerous",
            "reckless",
            "fine",
            "ticket",
            "police check",
        ]

        # Check for emergency
        is_emergency = any(keyword in query for keyword in emergency_keywords)
        is_emergency |= any(
            phrase in query for phrase in ["right now", "happening now", "currently", "immediately"]
        )

        # Determine category
        category = LegalCategory.GENERAL
        if any(keyword in query for keyword in traffic_keywords):
            category = LegalCategory.ROAD_TRAFFIC
        elif any(keyword in query for keyword in criminal_keywords):
            category = LegalCategory.CRIMINAL

        # Determine intent
        intent = QueryCategory.QUESTION
        if "?" in query or any(word in query for word in ["what", "how", "can", "is", "do"]):
            intent = QueryCategory.QUESTION
        elif any(phrase in query for phrase in ["not sure", "clarify", "what do you mean"]):
            intent = QueryCategory.CLARIFICATION

        # Determine urgency
        if is_emergency:
            urgency = UrgencyLevel.CRITICAL if "arrest" in query else UrgencyLevel.HIGH
        elif any(word in query for word in ["urgent", "asap", "soon"]):
            urgency = UrgencyLevel.HIGH
        else:
            urgency = UrgencyLevel.MEDIUM if category != LegalCategory.GENERAL else UrgencyLevel.LOW

        return {
            "intent": intent,
            "category": category,
            "urgency": urgency,
            "is_emergency": is_emergency,
            "confidence": 0.75,
            "reasoning": "Keyword-based classification",
        }


class LanguageDetectionAgent(BaseAgent):
    """
    Detect the language of the user query.
    Supports English and Twi (future).
    """

    @property
    def name(self) -> str:
        return "language_detection"

    @property
    def description(self) -> str:
        return "Detects the language of user queries"

    async def process(self, context: AgentContext) -> AgentResult:
        """
        Detect language from query.

        Args:
            context: Agent context

        Returns:
            AgentResult with language data
        """
        query = context.query

        # Basic Twi detection (can be improved)
        twi_indicators = [
            "sɛ",
            "woho",
            "meda",
            "akyiwade",
            "mmerɛ",
            "deɛ",
            "ɛna",
            "yɛ",
            "kɔ",
            "fi",
            "mu",
            "no",
            "ko",
        ]

        has_twi = any(indicator in query.lower() for indicator in twi_indicators)

        language = "twi" if has_twi else "english"

        context.metadata["language"] = language

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={"language": language},
            confidence=0.9 if language == "english" else 0.7,
        )
