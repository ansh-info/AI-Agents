"""
Test suite for Talk2Competitors system
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.main_agent import get_app, make_supervisor_node
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2competitors import Talk2Competitors
from ..tools.s2.search import search_tool
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations

# Mock response data
MOCK_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "123",
            "title": "Deep Learning",
            "abstract": "A study on deep learning",
            "year": 2023,
            "citationCount": 100,
            "url": "https://example.com/paper1",
        },
        {
            "paperId": "456",
            "title": "Machine Learning Applications",
            "abstract": "Applications of ML",
            "year": 2023,
            "citationCount": 50,
            "url": "https://example.com/paper2",
        },
    ]
}

MOCK_REC_RESPONSE = {
    "recommendedPapers": [
        {
            "paperId": "789",
            "title": "Neural Networks",
            "abstract": "Study on neural networks",
            "year": 2023,
            "citationCount": 75,
            "url": "https://example.com/paper3",
        }
    ]
}

# Unit Tests


@pytest.fixture
def base_state():
    """Base state fixture for tests"""
    return {"messages": [], "papers": {}, "is_last_step": False, "current_agent": None}


class TestSupervisorNode:
    """Unit tests for supervisor node functionality"""

    def test_supervisor_routes_to_s2_when_search_keyword(self):
        """Test that supervisor routes to S2 agent when search keyword is present"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Test response")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="search for machine learning papers")],
            papers={},
            is_last_step=False,
            current_agent=None,
        )

        result = supervisor(state)
        assert result.goto == "s2_agent"
        assert result.update["current_agent"] == "s2_agent"

    def test_supervisor_routes_to_end_for_general_query(self):
        """Test that supervisor routes to END for general queries"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Paris is the capital")

        supervisor = make_supervisor_node(llm_mock)
        state = Talk2Competitors(
            messages=[HumanMessage(content="what is the capital of France?")],
            papers={},
            is_last_step=False,
            current_agent=None,
        )

        result = supervisor(state)
        assert result.goto == "__end__"
        assert result.update["is_last_step"] == True


class TestS2Tools:
    """Unit tests for Semantic Scholar tools"""

    @patch("requests.get")
    def test_search_tool(self, mock_get):
        """Test search tool returns expected format and content"""
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        result = search_tool(query="machine learning", tool_call_id="test123", limit=2)

        papers = result.update["papers"]
        assert len(papers) == 2
        assert "Deep Learning" in papers["123"]["Title"]
        assert all(
            key in papers["123"]
            for key in ["Title", "Abstract", "Year", "Citation Count", "URL"]
        )

    @patch("requests.get")
    def test_single_paper_recommendations(self, mock_get):
        """Test single paper recommendations returns expected format"""
        mock_get.return_value.json.return_value = MOCK_REC_RESPONSE
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations(
            paper_id="123", tool_call_id="test123", limit=1
        )

        papers = result.update["papers"]
        assert len(papers) == 1
        assert "Neural Networks" in papers["789"]["Title"]

    @patch("requests.post")
    def test_multi_paper_recommendations(self, mock_post):
        """Test multi paper recommendations returns expected format"""
        mock_post.return_value.json.return_value = MOCK_REC_RESPONSE
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations(
            paper_ids=["123", "456"], tool_call_id="test123", limit=1
        )

        papers = result.update["papers"]
        assert len(papers) == 1
        assert "Neural Networks" in papers["789"]["Title"]


# Integration Tests


@pytest.mark.integration
def test_end_to_end_paper_search():
    """
    Integration test for complete paper search workflow
    Tests:
    1. Initial search
    2. Getting recommendations
    3. State management throughout
    """
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Setup mocks
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_post.return_value.json.return_value = MOCK_REC_RESPONSE
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200

        # Initialize app
        app = get_app("test_integration")

        # Test search
        response = app.invoke(
            {
                "messages": [
                    HumanMessage(content="search for machine learning papers")
                ],
                "papers": {},
                "is_last_step": False,
                "current_agent": None,
            }
        )

        assert len(response["papers"]) > 0
        assert "Deep Learning" in response["messages"][-1].content

        # Test recommendations
        response = app.invoke(
            {
                "messages": [
                    HumanMessage(content="get recommendations for the first paper")
                ],
                "papers": response["papers"],
                "is_last_step": False,
                "current_agent": None,
            }
        )

        assert len(response["papers"]) > 0
        assert "Neural Networks" in response["messages"][-1].content
