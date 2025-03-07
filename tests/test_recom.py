import unittest
from dotenv import load_dotenv
from agents.main_agent import get_app
from state.shared_state import Talk2Papers


class TestPaperRecommendations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        load_dotenv()
        cls.app = get_app("test_session")
        cls.base_state = Talk2Papers(messages=[], papers=[], search_table="")

    def test_search_and_single_recommendation(self):
        """Test workflow: search -> single paper recommendation"""
        # Step 1: Search for papers
        search_state = self.base_state.copy()
        search_state.messages = [
            {"role": "user", "content": "Search for papers about transformers in nlp"}
        ]

        search_result = self.app.invoke(search_state)

        # Verify search results
        self.assertTrue(search_result.get("papers"), "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract first paper ID
        # Assuming papers is a markdown table, parse to get paper ID
        paper_lines = papers.split("\n")
        if len(paper_lines) > 2:  # Header + separator + at least one paper
            paper_id = paper_lines[2].split("|")[1].strip()

            # Step 2: Get recommendations for first paper
            rec_state = self.base_state.copy()
            rec_state.messages = [
                {"role": "user", "content": f"Get recommendations for paper {paper_id}"}
            ]

            rec_result = self.app.invoke(rec_state)
            self.assertTrue(
                rec_result.get("papers"), "Should return paper recommendations"
            )
            print("Single Paper Recommendations:", rec_result.get("papers"))

    def test_search_and_multi_recommendation(self):
        """Test workflow: search -> multi paper recommendations"""
        # Step 1: Search for papers
        search_state = self.base_state.copy()
        search_state.messages = [
            {"role": "user", "content": "Search for papers about deep learning"}
        ]

        search_result = self.app.invoke(search_state)

        # Verify search results
        self.assertTrue(search_result.get("papers"), "Search should return papers")
        papers = search_result.get("papers")
        print("Search Results:", papers)

        # Extract first two paper IDs
        paper_lines = papers.split("\n")
        if len(paper_lines) > 3:  # Header + separator + at least two papers
            paper_id1 = paper_lines[2].split("|")[1].strip()
            paper_id2 = paper_lines[3].split("|")[1].strip()

            # Step 2: Get recommendations for both papers
            rec_state = self.base_state.copy()
            rec_state.messages = [
                {
                    "role": "user",
                    "content": f"Get recommendations for papers {paper_id1} and {paper_id2}",
                }
            ]

            rec_result = self.app.invoke(rec_state)
            self.assertTrue(
                rec_result.get("papers"), "Should return paper recommendations"
            )
            print("Multi Paper Recommendations:", rec_result.get("papers"))

    def test_invalid_paper_id(self):
        """Test handling of invalid paper ID"""
        invalid_state = self.base_state.copy()
        invalid_state.messages = [
            {"role": "user", "content": "Get recommendations for paper invalid_id_123"}
        ]

        result = self.app.invoke(invalid_state)
        self.assertIn("error", str(result).lower(), "Should handle invalid paper ID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
