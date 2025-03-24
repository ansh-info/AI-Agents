"""Test cases for Talk2Papers agents and tools"""

import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool, ToolException
from langchain_openai import ChatOpenAI
from agents.main_agent import get_app
from agents.s2_agent import s2_agent, SemanticScholarAgent
from tools.s2.search import search_tool
from tools.s2.single_paper_rec import get_single_paper_recommendations
from tools.s2.multi_paper_rec import get_multi_paper_recommendations
from tools.s2.display_results import display_results

# Mock data for tests
MOCK_PAPER_RESPONSE = {
    "recommendedPapers": [{"paperId": "abc123", "title": "Test Paper"}]
}


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

    assert response["current_agent"] == "s2_agent"
    assert isinstance(response["messages"][-1], AIMessage)


def test_main_agent_error_handling():
    """Test main agent error handling"""
    unique_id = "test_error"
    app = get_app(unique_id)

    # Test error in s2_agent
    with patch("agents.s2_agent.s2_agent.invoke", side_effect=Exception("Test error")):
        response = app.invoke(
            {
                "messages": [HumanMessage(content="Find papers about AI")],
                "papers": [],
                "is_last_step": False,
                "current_agent": None,
            },
            config={"configurable": {"thread_id": unique_id}},
        )
        assert "Error:" in response["messages"][-1].content


def test_s2_agent():
    """Test the S2 agent's functionality"""
    assert hasattr(s2_agent, "tools_agent")

    state = {
        "messages": [HumanMessage(content="Find papers about machine learning")],
        "papers": [],
        "is_last_step": False,
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = s2_agent.invoke(state)
        assert "messages" in response
        assert "papers" in response


def test_s2_agent_error_handling():
    """Test S2 agent error handling"""
    # Test initialization error
    with patch("langchain_openai.ChatOpenAI", side_effect=Exception("Init error")):
        with pytest.raises(Exception) as exc_info:
            SemanticScholarAgent()
        assert "Init error" in str(exc_info.value)

    # Test tool execution error
    state = {
        "messages": [HumanMessage(content="Find papers")],
        "papers": [],
        "is_last_step": False,
    }

    with patch(
        "langgraph.prebuilt.create_react_agent", side_effect=Exception("Tool error")
    ):
        response = s2_agent.invoke(state)
        assert "Error:" in response["messages"][0].content


def test_search_tool():
    """Test the search papers tool"""
    query = "machine learning"
    tool_call_id = "test_123"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPER_RESPONSE]}

        response = search_tool.func(query=query, tool_call_id=tool_call_id, limit=2)

        assert "papers" in response
        assert "messages" in response
        assert isinstance(response["papers"], list)


def test_search_edge_cases():
    """Test search tool edge cases"""
    # Test no results
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": []}
        response = search_tool.func(query="nonexistent topic", tool_call_id="test_123")
        assert "No papers found" in response["messages"][0].content

    # Test retry logic
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            type("obj", (object,), {"status_code": 429}),
            type("obj", (object,), {"status_code": 429}),
            type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: {"data": [{"paperId": "123", "title": "Test"}]},
                },
            ),
        ]
        response = search_tool.func(query="retry test", tool_call_id="test_123")
        assert len(response["papers"]) > 0


def test_single_paper_rec():
    """Test single paper recommendations"""
    paper_id = "1234567890123456789012345678901234567890"
    tool_call_id = "test_123"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = get_single_paper_recommendations.func(
            paper_id=paper_id, tool_call_id=tool_call_id, limit=2
        )

        assert response is not None
        assert "papers" in response.update


def test_multi_paper_rec():
    """Test multi paper recommendations"""
    paper_ids = [
        "1234567890123456789012345678901234567890",
        "0987654321098765432109876543210987654321",
    ]
    tool_call_id = "test_123"

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = get_multi_paper_recommendations.func(
            paper_ids=paper_ids, tool_call_id=tool_call_id, limit=2
        )

        assert response is not None
        assert "papers" in response.update


def test_multi_paper_rec_edge_cases():
    """Test multi paper recommendations edge cases"""
    paper_ids = [
        "1234567890123456789012345678901234567890",
        "0987654321098765432109876543210987654321",
    ]
    tool_call_id = "test_123"

    # Test 404 response
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 404
        response = get_multi_paper_recommendations.func(
            paper_ids=paper_ids, tool_call_id=tool_call_id
        )
        assert "paper IDs not found" in response.update["messages"][0].content

    # Test empty recommendations
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"recommendedPapers": []}
        response = get_multi_paper_recommendations.func(
            paper_ids=paper_ids, tool_call_id=tool_call_id
        )
        assert "No recommendations found" in response.update["messages"][0].content


def test_display_results():
    """Test display results tool"""
    state = {"papers": ["Paper 1", "Paper 2"], "messages": []}
    result = display_results(state)
    assert result == state["papers"]

    # Test with empty state
    empty_state = {"papers": []}
    result = display_results(empty_state)
    assert result == []


def test_error_handling():
    """Test error handling in tools"""
    # Test invalid paper ID format
    with pytest.raises((ValueError, ToolException)) as exc_info:
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 404
            get_single_paper_recommendations.func(
                paper_id="invalid_id", tool_call_id="test_123", limit=2
            )
    assert any(
        msg in str(exc_info.value)
        for msg in ["40-character hexadecimal", "Error getting recommendations"]
    )

    # Test too many paper IDs
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        with pytest.raises(ValueError) as exc_info:
            get_multi_paper_recommendations.func(
                paper_ids=["a" * 40 for _ in range(11)],  # 11 IDs
                tool_call_id="test_123",
                limit=2,
            )
    assert "Maximum of 10 paper IDs allowed" in str(exc_info.value)
