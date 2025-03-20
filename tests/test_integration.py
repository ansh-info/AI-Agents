"""
Unit and integration tests for Talk2Competitors system.
Each test focuses on a single piece of functionality.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from ..agents.main_agent import get_app, make_supervisor_node
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2competitors import Talk2Competitors
from ..tools.s2.search import search_tool
from ..tools.s2.display_results import display_results
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations

# Fixed test data for deterministic results
MOCK_PAPER = {
    "paperId": "123",
    "title": "Machine Learning Basics",
    "abstract": "An introduction to ML",
    "year": 2023,
    "citationCount": 100,
    "url": "https://example.com/paper1",
}


@pytest.fixture
def base_state():
    """Create a base state for tests"""
    return Talk2Competitors(
        messages=[],
        papers={},
        is_last_step=False,
        current_agent=None,
        llm_model="gpt-4",
    )


class TestMainAgentRouting:
    """
    Test the routing functionality of main agent.
    Verifies that queries are correctly routed to appropriate handlers.
    """

    def test_routes_search_query_to_s2(self):
        """Test that search queries are routed to S2 agent"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Search results")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="search for ML papers")],
            papers={},
            is_last_step=False,
            current_agent=None,
            llm_model="gpt-4",
        )

        result = supervisor(state)
        assert result.goto == "s2_agent"
        assert result.update["current_agent"] == "s2_agent"

    def test_routes_general_query_to_end(self):
        """Test that non-search queries end the conversation"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="General response")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="What is ML?")],
            papers={},
            is_last_step=False,
            current_agent=None,
            llm_model="gpt-4",
        )

        result = supervisor(state)
        assert result.goto == "__end__"
        assert result.update["is_last_step"] == True


class TestS2AgentFunctionality:
    """
    Test S2 agent's ability to process requests and manage state.
    """

    @patch("requests.get")
    def test_processes_search_request(self, mock_get):
        """Test S2 agent's handling of search requests"""
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPER]}
        mock_get.return_value.status_code = 200

        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Search completed")

        with patch("langchain_openai.ChatOpenAI", return_value=llm_mock):
            app = get_s2_app("test_id")
            state = Talk2Competitors(
                messages=[HumanMessage(content="search ML papers")],
                papers={},
                is_last_step=False,
                current_agent="s2_agent",
                llm_model="gpt-4",
            )

            response = app.invoke(state, {"configurable": {"thread_id": "test_id"}})
            assert len(response["messages"]) > 0


class TestS2Tools:
    """
    Test individual S2 tools functionality.
    Each test verifies a specific tool's behavior.
    """

    def test_display_results_shows_papers(self):
        """Test display_results tool shows papers from state"""
        mock_state = {"papers": {"123": MOCK_PAPER}}
        result = display_results.invoke({"input": mock_state})
        assert "123" in result
        assert result["123"]["title"] == "Machine Learning Basics"

    @patch("requests.get")
    def test_search_finds_papers(self, mock_get):
        """Test search tool finds and formats papers"""
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPER]}
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            {"query": "machine learning", "limit": 1, "tool_call_id": "test123"}
        )

        assert "papers" in result.update
        assert len(result.update["messages"]) == 1

    @patch("requests.get")
    def test_single_paper_recommendations(self, mock_get):
        """Test single paper recommendations tool"""
        mock_get.return_value.json.return_value = {"recommendedPapers": [MOCK_PAPER]}
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            {"paper_id": "123", "limit": 1, "tool_call_id": "test123"}
        )

        assert "papers" in result.update
        assert len(result.update["papers"]) > 0

    @patch("requests.post")
    def test_multi_paper_recommendations(self, mock_post):
        """Test multi-paper recommendations tool"""
        mock_post.return_value.json.return_value = {"recommendedPapers": [MOCK_PAPER]}
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            {"paper_ids": ["123", "456"], "limit": 1, "tool_call_id": "test123"}
        )

        assert "papers" in result.update
        assert len(result.update["papers"]) > 0


def test_end_to_end_search_workflow(base_state):
    """
    Integration test: Complete search workflow
    Tests the entire system working together.
    """
    with patch("requests.get") as mock_get, patch(
        "langchain_openai.ChatOpenAI"
    ) as mock_llm:
        # Setup mocks
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPER]}
        mock_get.return_value.status_code = 200

        llm_instance = Mock()
        llm_instance.invoke.return_value = AIMessage(content="Search results")
        mock_llm.return_value = llm_instance

        # Initialize and run
        app = get_app("test_integration")
        base_state.messages = [HumanMessage(content="search for ML papers")]

        config = {
            "configurable": {
                "thread_id": "test_integration",
                "checkpoint_ns": "test",
                "checkpoint_id": "test123",
            }
        }

        response = app.invoke(base_state, config)
        assert "papers" in response
        assert len(response["messages"]) > 0
