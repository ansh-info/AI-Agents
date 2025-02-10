import os
import sys
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.main_agent import main_agent
from state.shared_state import shared_state
from config.config import config


def reset_shared_state():
    """Reset shared state between tests"""
    shared_state.state = {
        config.StateKeys.PAPERS: [],
        config.StateKeys.SELECTED_PAPERS: [],
        config.StateKeys.CURRENT_TOOL: None,
        config.StateKeys.CURRENT_AGENT: None,
        config.StateKeys.RESPONSE: None,
        config.StateKeys.ERROR: None,
        config.StateKeys.CHAT_HISTORY: [],
        config.StateKeys.USER_INFO: {},
        config.StateKeys.MEMORY: {},
    }


def run_test_case(message: str, test_name: str) -> Dict[str, Any]:
    """Run a single test case and return results"""
    print(f"\nTest Case: {test_name}")
    print("Query:", message)

    # Prepare initial state
    state = {
        "message": message,
        "response": None,
        "error": None,
    }

    # Create and run the workflow
    graph = main_agent.create_graph()
    result = graph.invoke(state)

    # Print results
    print("\nResponse:", result.get("response"))
    print("Error:", result.get("error"))
    print("\nShared State:")
    print("Papers found:", len(shared_state.get(config.StateKeys.PAPERS)))
    print("Current agent:", shared_state.get(config.StateKeys.CURRENT_AGENT))

    return result


def test_paper_search():
    """Test basic paper search functionality"""
    reset_shared_state()
    return run_test_case(
        "Find papers about machine learning and neural networks", "Basic Paper Search"
    )


def test_paper_recommendations():
    """Test paper recommendations functionality"""
    # First, ensure we have some papers in state
    papers = shared_state.get(config.StateKeys.PAPERS)
    if not papers:
        print("\nSkipping recommendations test - no papers available")
        return None

    paper_id = papers[0].get("paperId")
    if not paper_id:
        print("\nSkipping recommendations test - no valid paper ID found")
        return None

    return run_test_case(f"Find papers similar to {paper_id}", "Paper Recommendations")


def test_invalid_query():
    """Test handling of invalid or unclear queries"""
    reset_shared_state()
    return run_test_case("Please do something undefined", "Invalid Query Handling")


def main():
    print("Starting tests...")
    print(f"Project root: {project_root}")

    # Run tests
    search_result = test_paper_search()
    if search_result and not search_result.get("error"):
        recommendations_result = test_paper_recommendations()

    # Test error handling
    error_result = test_invalid_query()

    print("\nTest suite completed.")


if __name__ == "__main__":
    main()
