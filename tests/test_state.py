import os
import sys
from typing import Dict

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.workflow_manager import WorkflowManager
from state.agent_state import AgentStatus


def print_state_info(state, test_name):
    """Helper function to print state information"""
    print(f"\n=== {test_name} ===")
    print(f"Status: {state.status}")
    print(f"Current step: {state.current_step}")
    print("Messages:")
    for msg in state.memory.messages:
        print(f"- {msg['role']}: {msg['content']}")
    if state.error_message:
        print(f"Error: {state.error_message}")
    if state.next_steps:
        print(f"Next steps: {state.next_steps}")


def test_workflow():
    try:
        # Initialize workflow manager
        print("Debug: Starting test_workflow")
        print("Initializing WorkflowManager...")
        manager = WorkflowManager()
        print("Debug: WorkflowManager initialized")

        # Test 1: Basic command
        print("\nDebug: Starting Test 1")
        state = manager.process_command_external("Hello")
        print("Debug: Completed command processing")
        print_state_info(state, "Test 1: Basic command")

        # Test 2: Help command
        print("\nDebug: Starting Test 2")
        manager.reset_state()  # Reset state before new command
        state = manager.process_command_external("help")
        print_state_info(state, "Test 2: Help command")

        # Test 3: Search command
        print("\nDebug: Starting Test 3")
        manager.reset_state()  # Reset state before new command
        state = manager.process_command_external("search LLM papers")
        print_state_info(state, "Test 3: Search command")

        # Test 4: Search context with data
        print("\nDebug: Starting Test 4")
        print("\n=== Test 4: Search context with data ===")
        manager.reset_state()
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

    except Exception as e:
        print(f"\nDebug: An error occurred: {str(e)}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    print("Debug: Script starting")
    test_workflow()
    print("Debug: Script completed")
