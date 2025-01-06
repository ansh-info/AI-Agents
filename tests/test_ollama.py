import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import asyncio

from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient
from state.agent_state import AgentState, AgentStatus

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "llama3.2:1b"


def print_section(name: str):
    """Print a section header"""
    print(f"\n{'='*20} {name} {'='*20}")


def print_state_info(state: AgentState, title: str = "State Info"):
    """Helper to print state information"""
    print(f"\n--- {title} ---")
    print(f"Status: {state.status}")
    print(f"Current step: {state.current_step}")
    if state.error_message:
        print(f"Error: {state.error_message}")
    print("\nMessages:")
    for msg in state.memory.messages:
        print(f"- {msg['role']}: {msg['content']}")
    if state.search_context.query:
        print(f"\nSearch query: {state.search_context.query}")


async def verify_ollama_connection():
    """Verify connection to Ollama service"""
    print_section("Verifying Ollama Connection")
    client = OllamaClient()
    print(f"Connecting to Ollama at: {client.base_url}")

    try:
        # Test simple generation
        response = await client.generate(
            prompt="Say hello", model=MODEL_NAME, max_tokens=20
        )
        print(f"Test response: {response}")

        if response.strip():
            print("✓ Connection test successful")
            return True
        else:
            print("✗ Empty response received")
            return False

    except Exception as e:
        print(f"✗ Connection test failed: {str(e)}")
        return False


async def test_workflow_commands():
    """Test various commands through the workflow"""
    print_section("Testing Workflow Commands")

    manager = EnhancedWorkflowManager(model_name=MODEL_NAME)

    test_cases = [
        ("help", "help_displayed", None),
        ("search papers about LLMs", "search_initiated", "papers about llms"),
        ("search", "invalid_search", None),
        ("tell me about agents", "command_processed", None),
    ]

    success = True
    for command, expected_step, expected_query in test_cases:
        try:
            print(f"\nTesting command: '{command}'")
            state = await manager.process_command_async(command)
            print_state_info(state)

            # Verify command processing
            assert (
                state.status == AgentStatus.SUCCESS
            ), f"Expected SUCCESS status for command: {command}"
            assert (
                state.current_step == expected_step
            ), f"Expected step '{expected_step}' but got '{state.current_step}' for command: {command}"

            if expected_query:
                assert (
                    state.search_context.query == expected_query
                ), f"Expected query '{expected_query}' but got '{state.search_context.query}'"
            elif command.startswith("search "):
                assert state.memory.messages[-1]["content"].startswith(
                    "Initiating search for:"
                ), "Search command should update memory with search confirmation"

            # Verify messages
            assert (
                len(state.memory.messages) >= 2
            ), "Should have user command and system response"
            assert state.memory.messages[0]["role"] == "user"
            assert state.memory.messages[-1]["role"] == "system"

        except AssertionError as e:
            print(f"✗ Test failed for command '{command}': {str(e)}")
            success = False
        except Exception as e:
            print(f"✗ Unexpected error for command '{command}': {str(e)}")
            success = False

    if success:
        print("\n✓ Workflow command tests passed")
    return success


async def main():
    """Run the test suite"""
    print_section("Starting Integrated Test Suite")

    # First verify Ollama connection
    if not await verify_ollama_connection():
        print("Failed to connect to Ollama service. Please check if it's running.")
        return False

    # Run workflow tests
    success = await test_workflow_commands()

    print_section("Test Summary")
    print("Overall Status:", "✓ Passed" if success else "✗ Failed")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
