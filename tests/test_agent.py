import os
import sys
from typing import Any, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.main_agent import main_agent
from config.config import config
from state.shared_state import shared_state


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

    # Prepare initial state for main agent
    state = {
        "message": message,
        "response": None,
        "error": None,
    }

    # Create and run the main agent workflow
    graph = main_agent.create_graph()
    result = graph.invoke(state)

    # Print results
    print("\nMain Agent Decision:")
    print("Response:", result.get("response"))
    print("Error:", result.get("error"))
    print("\nFinal State:")
    print("Papers found:", len(shared_state.get(config.StateKeys.PAPERS)))
    print("Routed to agent:", shared_state.get(config.StateKeys.CURRENT_AGENT))

    return result


def test_paper_search():
    """Test paper search through main agent"""
    reset_shared_state()
    return run_test_case(
        "Find recent papers about machine learning and neural networks in computer vision",
        "Paper Search Routing",
    )


def test_search_with_year():
    """Test year-specific search through main agent"""
    reset_shared_state()
    return run_test_case(
        "Find papers about quantum computing published in 2023",
        "Year-Specific Search Routing",
    )


def test_invalid_query():
    """Test handling of invalid queries through main agent"""
    reset_shared_state()
    return run_test_case("Please do something undefined", "Invalid Query Routing")


def main():
    print("Starting Main Agent Tests...")
    print(f"Project root: {project_root}")

    # Run routing tests
    print("\n=== Testing Search Routing ===")
    search_result = test_paper_search()

    print("\n=== Testing Year Search Routing ===")
    year_search_result = test_search_with_year()

    print("\n=== Testing Invalid Query Handling ===")
    error_result = test_invalid_query()

    print("\nTest suite completed.")


if __name__ == "__main__":
    main()
