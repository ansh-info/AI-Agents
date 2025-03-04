import os
import sys
from typing import Any, Dict, List

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


def run_test_case(input_query: str, test_name: str) -> Dict[str, Any]:
    """Run a single test case and return results"""
    print(f"\nTest Case: {test_name}")
    print("=" * 50)
    print("Input Query:", input_query)

    # Reset state for clean test
    reset_shared_state()

    # Prepare initial state with messages
    state = {"messages": [{"role": "user", "content": input_query}]}

    # Process through main agent
    result = main_agent.invoke(state)

    # Print results
    print("\nResults:")
    print("-" * 50)
    if "error" in result:
        print("Error:", result["error"])
    else:
        # Get the last message content
        last_message = result.get("messages", [])[-1]
        print("Response:", last_message.content if last_message else "No response")

    print("\nFinal State:")
    print("-" * 50)
    print("Papers found:", len(shared_state.get(config.StateKeys.PAPERS)))
    print("Processed by:", shared_state.get(config.StateKeys.CURRENT_AGENT))
    print("Tool used:", shared_state.get(config.StateKeys.CURRENT_TOOL))
    print("=" * 50 + "\n")

    return result


def test_basic_paper_search():
    """Test basic paper search functionality"""
    return run_test_case(
        "Find recent papers about machine learning and neural networks in computer vision",
        "Basic Paper Search Test",
    )


def test_year_specific_search():
    """Test year-specific search functionality"""
    return run_test_case(
        "Find papers about quantum computing published in 2023",
        "Year-Specific Search Test",
    )


def test_single_paper_recommendation():
    """Test single paper recommendation functionality"""
    # First get a paper ID through search
    search_result = run_test_case(
        "Find a recent paper about transformers in NLP", "Get Paper ID Test"
    )

    # Get paper ID from the search results
    papers = shared_state.get(config.StateKeys.PAPERS)
    if papers:
        paper_id = papers[0].get("paperId")
        return run_test_case(
            f"Find papers similar to {paper_id}", "Single Paper Recommendation Test"
        )
    return {"error": "No papers found to test recommendations"}


def test_multi_paper_recommendation():
    """Test multiple paper recommendation functionality"""
    # First get paper IDs through search
    search_result = run_test_case(
        "Find recent papers about deep learning", "Get Paper IDs Test"
    )

    # Get multiple paper IDs
    papers = shared_state.get(config.StateKeys.PAPERS)
    if len(papers) >= 2:
        paper_id1 = papers[0].get("paperId")
        paper_id2 = papers[1].get("paperId")
        return run_test_case(
            f"Find papers similar to {paper_id1} and {paper_id2}",
            "Multi-Paper Recommendation Test",
        )
    return {"error": "Not enough papers found to test multi-recommendations"}


def test_domain_specific_search():
    """Test domain-specific search with enhanced terms"""
    return run_test_case(
        "Find latest papers about attention mechanisms in transformers",
        "Domain-Specific Search Test",
    )


def test_invalid_query():
    """Test handling of invalid or unclear queries"""
    return run_test_case("Please do something undefined", "Invalid Query Test")


def test_invalid_paper_id():
    """Test handling of invalid paper ID for recommendations"""
    return run_test_case(
        "Find papers similar to invalid_paper_id_123", "Invalid Paper ID Test"
    )


def test_chat_context():
    """Test chat context preservation"""
    # First query
    result1 = run_test_case("Find papers about deep learning", "Initial Query Test")

    # Follow-up query
    result2 = run_test_case(
        "Find similar papers to the first one you found", "Follow-up Query Test"
    )

    return result2


def main():
    """Run all tests"""
    print("Starting Agent Tests...")
    print("Project root:", project_root)

    # Basic search tests
    test_basic_paper_search()
    test_year_specific_search()
    test_domain_specific_search()

    # Recommendation tests
    test_single_paper_recommendation()
    test_multi_paper_recommendation()

    # Error handling tests
    test_invalid_query()
    test_invalid_paper_id()

    # Context tests
    test_chat_context()

    print("\nTest suite completed.")


if __name__ == "__main__":
    main()
