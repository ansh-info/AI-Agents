"""
Unit and integration tests for Talk2Competitors system.
Each test focuses on a single piece of functionality.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from ..agents.main_agent import get_app, make_supervisor_node
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2competitors import Talk2Competitors, replace_dict
from ..tools.s2.search import search_tool
from ..tools.s2.display_results import display_results
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations

# Fixed test data for deterministic results
MOCK_PAPER = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "URL": "https://example.com/paper1",
    }
}


@pytest.fixture
def base_state():
    """Create a base state for tests"""
    return {
        "messages": [],
        "papers": {},
        "is_last_step": False,
        "current_agent": None,
        "llm_model": "gpt-4o-mini",
    }


class TestMainAgentRouting:
    """Test the routing functionality of main agent"""

    def test_routes_search_query_to_s2(self, base_state):
        """Test that search queries are routed to S2 agent"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Search results")

        supervisor = make_supervisor_node(llm_mock)
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="search for papers")]

        result = supervisor(state)
        assert result.goto == "s2_agent"
        assert result.update["current_agent"] == "s2_agent"

    def test_routes_general_query_to_end(self, base_state):
        """Test that non-search queries end the conversation"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="General response")

        supervisor = make_supervisor_node(llm_mock)
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="What is ML?")]

        result = supervisor(state)
        assert result.goto == "__end__"
        assert result.update["is_last_step"] == True


class TestS2Tools:
    """Test individual S2 tools"""

    def test_display_results():
        """Test display results tool"""
        state = {"papers": ["Paper 1", "Paper 2"], "messages": []}
        result = display_results(state)
        assert result == state["papers"]

        # Test with empty state
        empty_state = {"papers": []}
        result = display_results(empty_state)
        assert result == []

    @patch("requests.get")
    def test_search_tool(self, mock_get):
        """Test search tool functionality"""
        mock_get.return_value.json.return_value = {
            "data": [{"paperId": "123", "title": "Machine Learning Basics"}]
        }
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            {
                "name": "search_tool",
                "args": {"query": "machine learning", "limit": 1},
                "type": "tool_call",
                "tool_call_id": "test123",
            }
        )

        assert "papers" in result.update
        assert len(result.update["messages"]) == 1
        assert isinstance(result.update["messages"][0], ToolMessage)

    @patch("requests.get")
    def test_single_paper_rec(self, mock_get):
        """Test single paper recommendations tool"""
        mock_get.return_value.json.return_value = {
            "recommendedPapers": [{"paperId": "123", "title": "ML Paper"}]
        }
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            {
                "name": "get_single_paper_recommendations",
                "args": {"paper_id": "123", "limit": 1},
                "type": "tool_call",
                "tool_call_id": "test123",
            }
        )

        assert "papers" in result.update
        assert isinstance(result.update["messages"][0], ToolMessage)

    @patch("requests.post")
    def test_multi_paper_rec(self, mock_post):
        """Test multi paper recommendations tool"""
        mock_post.return_value.json.return_value = {
            "recommendedPapers": [{"paperId": "123", "title": "ML Paper"}]
        }
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            {
                "name": "get_multi_paper_recommendations",
                "args": {"paper_ids": ["123", "456"], "limit": 1},
                "type": "tool_call",
                "tool_call_id": "test123",
            }
        )

        assert "papers" in result.update
        assert isinstance(result.update["messages"][0], ToolMessage)


def test_end_to_end_search_workflow(base_state):
    """Integration test: Complete search workflow"""
    with patch("requests.get") as mock_get, patch(
        "langchain_openai.ChatOpenAI"
    ) as mock_llm:
        # Setup mocks
        mock_get.return_value.json.return_value = {
            "data": [{"paperId": "123", "title": "ML Paper"}]
        }
        mock_get.return_value.status_code = 200

        llm_instance = Mock()
        llm_instance.invoke.return_value = AIMessage(content="Search results")
        mock_llm.return_value = llm_instance

        # Initialize app
        app = get_app("test_integration")

        # Prepare state
        test_state = base_state.copy()
        test_state["messages"] = [HumanMessage(content="search for ML papers")]

        config = {
            "configurable": {
                "thread_id": "test_integration",
                "checkpoint_ns": "test",
                "checkpoint_id": "test123",
            }
        }

        response = app.invoke(test_state, config)
        assert "papers" in response
        assert len(response["messages"]) > 0


def test_replace_dict():
    """Test state dictionary replacement function"""
    existing = {"key1": "value1", "key2": "value2"}
    new = {"key3": "value3"}
    result = replace_dict(existing, new)
    assert result == new


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
