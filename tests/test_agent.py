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
    print("=" * 50)
    print("Input Query:", message)

    # Prepare initial state for main agent
    state = {
        "message": message,
        "response": None,
        "error": None,
    }

    # Process through main agent
    graph = main_agent.create_graph()
    result = graph.invoke(state)

    # Print results
    print("\nResults:")
    print("-" * 50)
    if result.get("error"):
        print("Error:", result["error"])
    else:
        print(result.get("response", "No response"))

    print("\nFinal State:")
    print("-" * 50)
    print("Papers found:", len(shared_state.get(config.StateKeys.PAPERS)))
    print("Processed by:", shared_state.get(config.StateKeys.CURRENT_AGENT))
    print("=" * 50 + "\n")

    return result


def test_paper_search():
    """Test paper search routing and results"""
    reset_shared_state()
    return run_test_case(
        "Find recent papers about machine learning and neural networks in computer vision",
        "Basic Paper Search Test",
    )


def test_search_with_year():
    """Test year-specific search routing and results"""
    reset_shared_state()
    return run_test_case(
        "Find papers about quantum computing published in 2023",
        "Year-Specific Search Test",
    )


def test_invalid_query():
    """Test invalid query handling"""
    reset_shared_state()
    return run_test_case("Please do something undefined", "Invalid Query Test")


def main():
    print("Starting Agent Tests...")
    print("Project root:", project_root)

    test_paper_search()
    test_search_with_year()
    test_invalid_query()

    print("\nTest suite completed.")


if __name__ == "__main__":
    main()
