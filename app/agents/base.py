"""
Base Agent Class.
Defines the interface and common functionality for all agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class AgentStatus(str, Enum):
    """Agent execution status."""

    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class AgentContext:
    """Context passed between agents."""

    query: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    conversation_history: Optional[list] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.conversation_history is None:
            self.conversation_history = []


@dataclass
class AgentResult:
    """Result returned by an agent."""

    status: AgentStatus
    data: Dict[str, Any]
    message: Optional[str] = None
    next_agent: Optional[str] = None
    confidence: float = 0.0

    def is_success(self) -> bool:
        return self.status == AgentStatus.SUCCESS

    def is_failure(self) -> bool:
        return self.status == AgentStatus.FAILED

    def is_rejected(self) -> bool:
        return self.status == AgentStatus.REJECTED

    def should_escalate(self) -> bool:
        return self.status == AgentStatus.ESCALATED


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Each agent should:
    1. Have a unique name and description
    2. Implement the process method
    3. Return an AgentResult
    """

    def __init__(self):
        self._execution_count = 0
        self._execution_times = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description."""
        pass

    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Process the context and return a result.

        Args:
            context: Agent context containing query and metadata

        Returns:
            AgentResult with status and data
        """
        pass

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute the agent with timing and error handling.

        Args:
            context: Agent context

        Returns:
            AgentResult
        """
        start_time = datetime.now(timezone.utc)

        try:
            result = await self.process(context)
            self._execution_count += 1
            return result
        except Exception as e:
            # Log error and return failure result
            return AgentResult(
                status=AgentStatus.FAILED, data={}, message=f"Agent {self.name} failed: {str(e)}"
            )
        finally:
            # Track execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._execution_times.append(execution_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        avg_time = (
            sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        )

        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "avg_execution_time": avg_time,
        }


class AgentChain:
    """
    Chain multiple agents to execute in sequence.
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute all agents in sequence.

        Args:
            context: Initial context

        Returns:
            Final result from the chain
        """
        current_context = context
        final_result = None

        for agent in self.agents:
            result = await agent.execute(current_context)

            # Update context with agent data
            current_context.metadata.update(result.data)

            # Check if agent rejected or failed
            if result.is_rejected() or result.is_failure():
                return result

            # Check if agent specified next agent
            if result.next_agent:
                # Skip to specified agent (not implemented in simple chain)
                pass

            final_result = result

        return final_result


class AgentRouter:
    """
    Route queries to appropriate agents based on conditions.
    """

    def __init__(self):
        self._routes: Dict[str, BaseAgent] = {}
        self._default_agent: Optional[BaseAgent] = None

    def register_agent(self, condition: str, agent: BaseAgent, is_default: bool = False):
        """
        Register an agent for a condition.

        Args:
            condition: Condition string (e.g., "emergency", "criminal")
            agent: Agent to execute
            is_default: Whether this is the default agent
        """
        self._routes[condition] = agent
        if is_default:
            self._default_agent = agent

    async def route(self, condition: str, context: AgentContext) -> AgentResult:
        """
        Route to appropriate agent.

        Args:
            condition: Routing condition
            context: Agent context

        Returns:
            Agent result
        """
        agent = self._routes.get(condition, self._default_agent)

        if agent is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                data={},
                message=f"No agent found for condition: {condition}",
            )

        return await agent.execute(context)
