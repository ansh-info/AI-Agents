import os
import sys
from typing import Dict

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.workflow_manager import WorkflowManager
from state.agent_state import AgentStatus


def test_workflow():
    # Initialize workflow manager
    print("Initializing WorkflowManager...")
    manager = WorkflowManager()

    # Test 1: Basic command processing
    print("\nTest 1: Basic command processing")
    state = manager.process_command_external("Hello")
    print(f"Command Status: {state.status}")
    print(f"Current step: {state.current_step}")
    print(f"Messages: {state.memory.messages}")

    # Test 2: Help command
    print("\nTest 2: Help command")
    state = manager.process_command_external("help")
    print(f"Command Status: {state.status}")
    print(f"Messages: {state.memory.messages}")

    # Test 3: Search command placeholder
    print("\nTest 3: Search command")
    state = manager.process_command_external("search LLM papers")
    print(f"Command Status: {state.status}")
    print(f"Current step: {state.current_step}")

    # Test 4: Error handling
    print("\nTest 4: Error handling")
    manager.reset_state()  # Reset state for clean test
    try:
        raise Exception("Test error")
    except Exception as e:
        state = manager.get_state()
        state.update_status(AgentStatus.ERROR, str(e))
    print(f"Error Status: {state.status}")
    print(f"Error Message: {state.error_message}")

    # Test 5: Search context with data
    print("\nTest 5: Search context with data")
    manager.reset_state()  # Reset state for clean test
    sample_results = pd.DataFrame(
        {
            "title": ["Paper 1", "Paper 2"],
            "authors": ["Author 1", "Author 2"],
            "year": [2023, 2024],
        }
    )

    state = manager.get_state()
    state.update_search_results(sample_results, total_results=2)

    print(f"Search Status: {state.status}")
    if state.search_context.results is not None:
        print(f"Results shape: {state.search_context.results.shape}")
        print("Search Results:")
        print(state.search_context.results)


if __name__ == "__main__":
    test_workflow()
