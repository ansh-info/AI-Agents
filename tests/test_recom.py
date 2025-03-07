import unittest
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.main_agent import get_app
from state.shared_state import Talk2Papers


class TestPaperRecommendations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        load_dotenv()
        cls.app = get_app("test_session")

    def setUp(self):
        """Set up before each test"""
        self.base_state = Talk2Papers(messages=[], papers=[], search_table="")
        self.config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "checkpoint_ns": "test",
                "checkpoint_id": str(uuid.uuid4()),
            }
        }

    def extract_paper_id_from_table(self, table_str):
        """Helper method to extract paper ID from markdown table"""
        lines = table_str.split("\n")
        # Find the first data row (after header and separator)
        for line in lines:
            if (
                "|" in line
                and not "===" in line
                and not "---" in line
                and not "Paper ID" in line
            ):
                # Split by | and get the Paper ID column
                cols = line.split("|")
                if len(cols) >= 3:  # Make sure we have enough columns
                    return cols[
                        2
                    ].strip()  # Paper ID is in the second column (index 2 due to empty first split)
        return None

    def test_search_and_single_recommendation(self):
        """Test workflow: search -> single paper recommendation"""
        # Step 1: Search for papers
        search_message = HumanMessage(
            content="Search for papers about transformers in nlp"
        )
        search_state = Talk2Papers(
            messages=[search_message], papers=[], search_table=""
        )

        search_result = self.app.invoke(search_state, self.config)

        # Verify search results
        self.assertIn("papers", search_result, "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract paper ID using helper method
        paper_id = self.extract_paper_id_from_table(papers)
        self.assertIsNotNone(paper_id, "Failed to extract paper ID from search results")
        print(f"Extracted Paper ID: {paper_id}")

        # Step 2: Get recommendations for the paper
        rec_message = HumanMessage(content=f"Get recommendations for paper {paper_id}")
        rec_state = Talk2Papers(messages=[rec_message], papers=[], search_table="")

        rec_result = self.app.invoke(rec_state, self.config)
        self.assertIn("papers", rec_result, "Should return paper recommendations")
        print("Single Paper Recommendations:", rec_result.get("papers"))

    def test_search_and_multi_recommendation(self):
        """Test workflow: search -> multi paper recommendations"""
        # Step 1: Search for papers with higher limit
        search_message = HumanMessage(content="Search for 3 papers about deep learning")
        search_state = Talk2Papers(
            messages=[search_message], papers=[], search_table=""
        )

        search_result = self.app.invoke(search_state, self.config)

        # Verify search results
        self.assertIn("papers", search_result, "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract first paper ID
        paper_id1 = self.extract_paper_id_from_table(papers)
        self.assertIsNotNone(paper_id1, "Failed to extract first paper ID")

        # Search for another paper to get second ID
        search_message2 = HumanMessage(
            content="Search for papers about neural networks"
        )
        search_state2 = Talk2Papers(
            messages=[search_message2], papers=[], search_table=""
        )

        search_result2 = self.app.invoke(search_state2, self.config)
        paper_id2 = self.extract_paper_id_from_table(search_result2.get("papers"))
        self.assertIsNotNone(paper_id2, "Failed to extract second paper ID")

        print(f"Using paper IDs: {paper_id1} and {paper_id2}")

        # Step 2: Get recommendations for both papers
        rec_message = HumanMessage(
            content=f"Get recommendations for papers {paper_id1} and {paper_id2}"
        )
        rec_state = Talk2Papers(messages=[rec_message], papers=[], search_table="")

        rec_result = self.app.invoke(rec_state, self.config)
        self.assertIn("papers", rec_result, "Should return paper recommendations")
        print("Multi Paper Recommendations:", rec_result.get("papers"))

    def test_invalid_paper_id(self):
        """Test handling of invalid paper ID"""
        invalid_message = HumanMessage(
            content="Get recommendations for paper invalid_id_123"
        )
        invalid_state = Talk2Papers(
            messages=[invalid_message], papers=[], search_table=""
        )

        result = self.app.invoke(invalid_state, self.config)
        # Check if error message is in the response
        messages = result.get("messages", [])
        response_text = " ".join([str(m.content) for m in messages])
        self.assertIn("error", response_text.lower(), "Should handle invalid paper ID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
