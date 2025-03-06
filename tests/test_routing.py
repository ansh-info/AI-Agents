import unittest
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from agents.main_agent import get_app
from langchain_core.messages import HumanMessage


class TestTalk2Papers(unittest.TestCase):
    def setUp(self):
        self.app = get_app(uniq_id=1234)  # Changed 'unique_id' to 'uniq_id'

    def test_search_query(self):
        """Test basic paper search"""
        query = "machine learning in healthcare"
        response = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": 1234}},
        )
        self.assertIn("messages", response)

    def test_paper_recommendations(self):
        """Test paper recommendations"""
        query = "machine learning in healthcare"
        response = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": 1234}},
        )

        # Get recommendations for first paper
        paper_id = "SAMPLE_PAPER_ID"  # We should extract this from the first response
        query = f"Find similar papers to {paper_id}"
        response = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": 1234}},
        )
        self.assertIn("messages", response)

    def test_routing(self):
        """Test routing between main agent and s2 agent"""
        queries = [
            "search for papers about machine learning",
            "find similar papers to the first result",
            "what are the main topics in these papers",
        ]

        for query in queries:
            response = self.app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": 1234}},
            )
            self.assertIn("messages", response)


if __name__ == "__main__":
    unittest.main()
