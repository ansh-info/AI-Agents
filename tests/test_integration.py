"""
Test suite for Talk2Competitors system
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from ..agents.main_agent import get_app, make_supervisor_node
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2competitors import Talk2Competitors
from ..tools.s2.search import search_tool, SearchInput
from ..tools.s2.display_results import display_results
from ..tools.s2.single_paper_rec import (
    get_single_paper_recommendations,
    SinglePaperRecInput,
)
from ..tools.s2.multi_paper_rec import (
    get_multi_paper_recommendations,
    MultiPaperRecInput,
)

# Mock data
MOCK_PAPERS = {
    "123": {
        "Title": "Deep Learning",
        "Abstract": "A study on deep learning",
        "Year": 2023,
        "Citation Count": 100,
        "URL": "https://example.com/paper1",
    }
}


class TestMainAgent:
    """Test main agent routing functionality"""

    def test_supervisor_routes_to_s2(self):
        """Test routing to S2 agent for search queries"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Test response")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="search for papers")],
            papers={},
            is_last_step=False,
            current_agent=None,
            llm_model="test-model",
        )

        result = supervisor(state)
        assert result.goto == "s2_agent"
        assert result.update["current_agent"] == "s2_agent"

    def test_supervisor_routes_to_end(self):
        """Test routing to END for general queries"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Paris")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="capital of France")],
            papers={},
            is_last_step=False,
            current_agent=None,
            llm_model="test-model",
        )

        result = supervisor(state)
        assert result.goto == "__end__"
        assert result.update["is_last_step"] == True


class TestS2Agent:
    """Test S2 agent functionality"""

    @patch("requests.get")
    def test_s2_agent_workflow(self, mock_get):
        """Test S2 agent workflow with search query"""
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPERS["123"]]}
        mock_get.return_value.status_code = 200

        app = get_s2_app("test_id")
        state = Talk2Competitors(
            messages=[HumanMessage(content="search for ML papers")],
            papers={},
            is_last_step=False,
            current_agent="s2_agent",
            llm_model="test-model",
        )

        response = app.invoke(state, {"configurable": {"thread_id": "test_id"}})
        assert isinstance(response["messages"][-1], ToolMessage)


class TestS2Tools:
    """Test individual S2 tools"""

    def test_display_results(self):
        """Test display_results tool"""
        state = {"papers": MOCK_PAPERS}
        result = display_results.invoke(state=state)
        assert result == MOCK_PAPERS

    @patch("requests.get")
    def test_search_tool(self, mock_get):
        """Test search tool functionality"""
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPERS["123"]]}
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            SearchInput(query="machine learning", limit=1, tool_call_id="test123")
        )
        assert "papers" in result.update
        assert isinstance(result.update["messages"][0], ToolMessage)

    @patch("requests.get")
    def test_single_paper_rec(self, mock_get):
        """Test single paper recommendations"""
        mock_get.return_value.json.return_value = {
            "recommendedPapers": [MOCK_PAPERS["123"]]
        }
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            SinglePaperRecInput(paper_id="123", limit=1, tool_call_id="test123")
        )
        assert "papers" in result.update
        assert isinstance(result.update["messages"][0], ToolMessage)

    @patch("requests.post")
    def test_multi_paper_rec(self, mock_post):
        """Test multi paper recommendations"""
        mock_post.return_value.json.return_value = {
            "recommendedPapers": [MOCK_PAPERS["123"]]
        }
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            MultiPaperRecInput(
                paper_ids=["123", "456"], limit=1, tool_call_id="test123"
            )
        )
        assert "papers" in result.update
        assert isinstance(result.update["messages"][0], ToolMessage)


@pytest.mark.integration
def test_end_to_end_workflow():
    """Integration test for complete workflow"""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Setup mocks
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPERS["123"]]}
        mock_get.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "recommendedPapers": [MOCK_PAPERS["123"]]
        }
        mock_post.return_value.status_code = 200

        # Initialize app
        app = get_app("test_integration", "test-model")

        # Test complete workflow
        config = {
            "configurable": {
                "thread_id": "test_integration",
                "checkpoint_ns": "test",
                "checkpoint_id": "test123",
            }
        }

        response = app.invoke(
            {
                "messages": [HumanMessage(content="search for ML papers")],
                "papers": {},
                "is_last_step": False,
                "current_agent": None,
                "llm_model": "test-model",
            },
            config=config,
        )

        assert "papers" in response
        assert len(response["messages"]) > 0
