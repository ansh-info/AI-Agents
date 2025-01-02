import os
import sys
from typing import Dict

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from state.base_state import AgentState, AgentStatus, WorkflowManager


def test_workflow():
    # Initialize workflow manager
    manager = WorkflowManager()

    # Test 1: Basic command processing
    print("\nTest 1: Basic command processing")
    state = manager.process_command("Hello")
    print(f"Status: {state.status}")
    print(f"Current step: {state.current_step}")
    print(f"Message history: {state.memory.messages}")

    # Test 2: Error handling
    print("\nTest 2: Error handling")
    try:
        # Simulate an error
        raise Exception("Test error")
    except Exception as e:
        state.update_status(AgentStatus.ERROR, str(e))
    print(f"Status: {state.status}")
    print(f"Error message: {state.error_message}")

    # Test 3: Search context
    print("\nTest 3: Search context")
    # Create sample search results
    sample_results = pd.DataFrame(
        {
            "title": ["Paper 1", "Paper 2"],
            "authors": ["Author 1", "Author 2"],
            "year": [2023, 2024],
        }
    )
    state.update_search_results(sample_results, total_results=2)
    print(f"Search results shape: {state.search_context.results.shape}")
    print(f"Total results: {state.search_context.total_results}")


if __name__ == "__main__":
    test_workflow()
