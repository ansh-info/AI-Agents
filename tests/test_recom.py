import unittest
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

    def test_search_and_single_recommendation(self):
        """Test workflow: search -> single paper recommendation"""
        # Step 1: Search for papers
        search_message = HumanMessage(
            content="Search for papers about transformers in nlp"
        )
        search_state = Talk2Papers(
            messages=[search_message], papers=[], search_table=""
        )

        search_result = self.app.invoke(search_state)

        # Verify search results
        self.assertIn("papers", search_result, "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract first paper ID
        # Assuming papers is a markdown table, parse to get paper ID
        paper_lines = papers.split("\n")
        if len(paper_lines) > 2:  # Header + separator + at least one paper
            paper_id = paper_lines[2].split("|")[1].strip()

            # Step 2: Get recommendations for first paper
            rec_message = HumanMessage(
                content=f"Get recommendations for paper {paper_id}"
            )
            rec_state = Talk2Papers(messages=[rec_message], papers=[], search_table="")

            rec_result = self.app.invoke(rec_state)
            self.assertIn("papers", rec_result, "Should return paper recommendations")
            print("Single Paper Recommendations:", rec_result.get("papers"))

    def test_search_and_multi_recommendation(self):
        """Test workflow: search -> multi paper recommendations"""
        # Step 1: Search for papers
        search_message = HumanMessage(content="Search for papers about deep learning")
        search_state = Talk2Papers(
            messages=[search_message], papers=[], search_table=""
        )

        search_result = self.app.invoke(search_state)

        # Verify search results
        self.assertIn("papers", search_result, "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract first two paper IDs
        paper_lines = papers.split("\n")
        if len(paper_lines) > 3:  # Header + separator + at least two papers
            paper_id1 = paper_lines[2].split("|")[1].strip()
            paper_id2 = paper_lines[3].split("|")[1].strip()

            # Step 2: Get recommendations for both papers
            rec_message = HumanMessage(
                content=f"Get recommendations for papers {paper_id1} and {paper_id2}"
            )
            rec_state = Talk2Papers(messages=[rec_message], papers=[], search_table="")

            rec_result = self.app.invoke(rec_state)
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

        result = self.app.invoke(invalid_state)
        # Check if error message is in the response
        messages = result.get("messages", [])
        response_text = " ".join([str(m.content) for m in messages])
        self.assertIn("error", response_text.lower(), "Should handle invalid paper ID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
