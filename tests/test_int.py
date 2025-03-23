"""
Test cases for Talk2Papers agents and tools
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from ..agents.main_agent import get_app
from ..agents.s2_agent import s2_agent
from ..tools.s2.search import search_tool
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations


def test_main_agent_routing():
    """Test the main agent's routing capabilities"""
    unique_id = "test_12345"
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}

    # Test search routing
    prompt = "Find me papers about machine learning"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    # Verify routing to S2 agent
    assert response["current_agent"] == "s2_agent"
    assert isinstance(response["messages"][-1], AIMessage)

    # Test non-search routing
    prompt = "Thank you for your help"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    # Verify direct response without routing
    assert response["is_last_step"] == True
    assert response["current_agent"] is None


def test_s2_agent():
    """Test the S2 agent's functionality"""
    # Test initialization
    assert isinstance(s2_agent.tools_agent, Tool)

    # Test invocation
    state = {
        "messages": [HumanMessage(content="Find papers about machine learning")],
        "papers": [],
        "is_last_step": False,
    }

    response = s2_agent.invoke(state)
    assert "messages" in response
    assert "papers" in response
    assert isinstance(response["messages"][-1], AIMessage)


def test_search_tool():
    """Test the search papers tool"""
    query = "machine learning"
    tool_call_id = "test_123"

    response = search_tool.func(query=query, tool_call_id=tool_call_id, limit=2)

    assert "papers" in response
    assert "messages" in response
    assert isinstance(response["papers"], list)
    assert isinstance(response["messages"][0], AIMessage)


def test_single_paper_rec():
    """Test single paper recommendations"""
    # Using a valid paper ID format
    paper_id = "1234567890123456789012345678901234567890"  # 40 chars
    tool_call_id = "test_123"

    response = get_single_paper_recommendations.func(
        paper_id=paper_id, tool_call_id=tool_call_id, limit=2
    )

    assert response is not None
    assert hasattr(response, "update")


def test_multi_paper_rec():
    """Test multi paper recommendations"""
    paper_ids = [
        "1234567890123456789012345678901234567890",
        "0987654321098765432109876543210987654321",
    ]
    tool_call_id = "test_123"

    response = get_multi_paper_recommendations.func(
        paper_ids=paper_ids, tool_call_id=tool_call_id, limit=2
    )

    assert response is not None
    assert hasattr(response, "update")


def test_error_handling():
    """Test error handling in tools"""
    # Test invalid paper ID format
    with pytest.raises(ValueError):
        get_single_paper_recommendations.func(
            paper_id="invalid_id", tool_call_id="test_123"
        )

    # Test empty paper IDs list
    with pytest.raises(ValueError):
        get_multi_paper_recommendations.func(paper_ids=[], tool_call_id="test_123")
