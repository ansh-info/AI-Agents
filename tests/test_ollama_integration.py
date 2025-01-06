import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import asyncio

import pytest

from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient, sync_generate
from state.agent_state import AgentState, AgentStatus


async def test_ollama_client_basic():
    """Test basic Ollama client functionality"""
    print("\n=== Testing Basic Ollama Client ===")
    client = OllamaClient()

    try:
        # Test simple generation
        response = await client.generate(
            prompt="What is artificial intelligence?", model="llama3.2:1b"
        )
        print(
            f"Basic generation test response: {client.extract_response(response)[:100]}..."
        )

        # Test with system prompt
        response = await client.generate(
            prompt="What is artificial intelligence?",
            model="llama3.2:1b",
            system_prompt="You are a computer science professor.",
        )
        print(
            f"System prompt test response: {client.extract_response(response)[:100]}..."
        )

        print("✓ Basic Ollama client tests passed")
        return True
    except Exception as e:
        print(f"✗ Ollama client test failed: {str(e)}")
        return False


def test_sync_generate():
    """Test synchronous generation wrapper"""
    print("\n=== Testing Sync Generation ===")
    try:
        response = sync_generate(
            prompt="Explain what an LLM is.",
            system_prompt="You are a helpful AI assistant.",
        )
        print(f"Sync generation response: {response[:100]}...")
        print("✓ Sync generation test passed")
        return True
    except Exception as e:
        print(f"✗ Sync generation test failed: {str(e)}")
        return False


async def test_enhanced_workflow():
    """Test enhanced workflow manager with LLM integration"""
    print("\n=== Testing Enhanced Workflow ===")
    manager = EnhancedWorkflowManager()

    test_cases = [
        ("search papers about LLMs", "search_initiated"),
        ("help", "help_displayed"),
        ("invalid command", "command_processed"),
    ]

    try:
        for command, expected_step in test_cases:
            print(f"\nTesting command: {command}")

            # Reset state for new command
            manager.reset_state()

            # Add command to state
            state = manager.get_state()
            state.add_message("user", command)

            # Process command
            result = await manager.process_command_async(command)

            # Print results
            print(f"Status: {result.status}")
            print(f"Current step: {result.current_step}")
            print(f"Expected step: {expected_step}")
            print("Messages:")
            for msg in result.memory.messages:
                print(f"- {msg['role']}: {msg['content']}")

            # Verify results
            assert (
                result.status == AgentStatus.SUCCESS
            ), f"Expected SUCCESS status, got {result.status}"
            assert (
                result.current_step == expected_step
            ), f"Expected step {expected_step}, got {result.current_step}"

        print("\n✓ Enhanced workflow tests passed")
        return True
    except Exception as e:
        print(f"\n✗ Enhanced workflow test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("\n=== Starting Ollama Integration Tests ===")

    # Test results
    results = {
        "ollama_client": await test_ollama_client_basic(),
        "sync_generate": test_sync_generate(),
        "enhanced_workflow": await test_enhanced_workflow(),
    }

    # Print summary
    print("\n=== Test Summary ===")
    for test_name, passed in results.items():
        print(f"{test_name}: {'✓ Passed' if passed else '✗ Failed'}")

    # Return overall success
    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
