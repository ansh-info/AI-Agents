import os
import sys
from typing import Dict

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.workflow_manager import WorkflowManager
from state.agent_state import AgentStatus
from state.base_state import AgentState, AgentStatus, WorkflowManager


def test_workflow():
    try:
        # Initialize workflow manager
        print("Initializing WorkflowManager...")
        manager = WorkflowManager()

        # Test 1: Basic command processing
        print("\nTest 1: Basic command processing")
        try:
            state = manager.process_command("Hello")
            print("Command processed successfully")
            print(f"Status: {state.status}")
            print(f"Current step: {state.current_step}")
            print(f"Message history: {state.memory.messages}")
        except Exception as e:
            print(f"Error in Test 1: {str(e)}")

        # Test 2: Error handling
        print("\nTest 2: Error handling")
        try:
            state = manager.get_state()
            state.update_status(AgentStatus.ERROR, "Test error")
            print("Error state updated successfully")
            print(f"Status: {state.status}")
            print(f"Error message: {state.error_message}")
        except Exception as e:
            print(f"Error in Test 2: {str(e)}")

        # Test 3: Search context
        print("\nTest 3: Search context")
        try:
            # Create sample search results
            sample_results = pd.DataFrame(
                {
                    "title": ["Paper 1", "Paper 2"],
                    "authors": ["Author 1", "Author 2"],
                    "year": [2023, 2024],
                }
            )

            print("Created sample DataFrame:")
            print(sample_results)

            state = manager.get_state()
            state.update_search_results(sample_results, total_results=2)

            print("\nAfter updating search results:")
            if state.search_context.results is not None:
                print(f"Search results shape: {state.search_context.results.shape}")
                print(f"Search results content:")
                print(state.search_context.results)
            else:
                print("Search results are None")

            print(f"Total results: {state.search_context.total_results}")

        except Exception as e:
            print(f"Error in Test 3: {str(e)}")

    except Exception as e:
        print(f"Global error: {str(e)}")


if __name__ == "__main__":
    test_workflow()
