import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.main_agent import main_agent
from state.shared_state import shared_state


def test_paper_search():
    """Test basic paper search functionality"""
    # Test case 1: Simple paper search
    state = {
        "message": "Find papers about machine learning and neural networks",
        "response": None,
        "error": None,
    }

    print("\nTest Case 1: Paper Search")
    print("Query:", state["message"])

    # Create and run the workflow
    graph = main_agent.create_graph()
    result = graph.invoke(state)

    print("\nResponse:", result.get("response"))
    print("Error:", result.get("error"))
    print("\nShared State:")
    print("Papers found:", len(shared_state.get("papers")))
    print("Current agent:", shared_state.get("current_agent"))


def test_paper_recommendations():
    """Test paper recommendations functionality"""
    # Test case 2: Paper recommendations
    # First, we need to have some papers in state
    if len(shared_state.get("papers")) > 0:
        paper_id = shared_state.get("papers")[0].get("paperId")
        state = {
            "message": f"Find papers similar to {paper_id}",
            "response": None,
            "error": None,
        }

        print("\nTest Case 2: Paper Recommendations")
        print("Query:", state["message"])

        graph = main_agent.create_graph()
        result = graph.invoke(state)

        print("\nResponse:", result.get("response"))
        print("Error:", result.get("error"))


def main():
    print("Starting tests...")
    print(f"Project root: {project_root}")

    # Clear shared state before testing
    shared_state.clear_response()

    # Run tests
    test_paper_search()
    test_paper_recommendations()


if __name__ == "__main__":
    main()
