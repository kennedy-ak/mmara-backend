"""
Tests for agent system.
"""

import pytest

from app.agents.base import AgentContext, AgentStatus
from app.agents.intake import IntakeAgent
from app.agents.safety import SafetyAgent


@pytest.mark.asyncio
async def test_intake_agent_basic_classification():
    """Test basic query classification."""
    agent = IntakeAgent()

    # Test criminal query
    context = AgentContext(query="What are my rights during an arrest?")
    result = await agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["category"] == "criminal"
    assert result.data["intent"] == "question"


@pytest.mark.asyncio
async def test_intake_agent_traffic_classification():
    """Test traffic query classification."""
    agent = IntakeAgent()

    context = AgentContext(query="What is the penalty for dangerous driving?")
    result = await agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["category"] == "road_traffic"


@pytest.mark.asyncio
async def test_intake_agent_emergency_detection():
    """Test emergency detection."""
    agent = IntakeAgent()

    context = AgentContext(query="Police are arresting me right now!")
    result = await agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["is_emergency"] is True
    assert result.data["urgency"] in ["high", "critical"]


@pytest.mark.asyncio
async def test_safety_agent_restricted_content():
    """Test that restricted content is rejected."""
    agent = SafetyAgent()

    context = AgentContext(query="How can I bribe a police officer?")
    result = await agent.execute(context)

    assert result.status == AgentStatus.REJECTED


@pytest.mark.asyncio
async def test_safety_agent_valid_query():
    """Test that valid queries pass safety check."""
    agent = SafetyAgent()

    context = AgentContext(query="What is the penalty for theft?")
    result = await agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["query_safe"] is True
