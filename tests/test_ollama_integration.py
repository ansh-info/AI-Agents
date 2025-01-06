import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import asyncio

import pytest

from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient, sync_generate
from state.agent_state import AgentState, AgentStatus

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "llama3.2:1b"


def print_section(name: str):
    """Print a section header"""
    print(f"\n{'='*20} {name} {'='*20}")


async def verify_ollama_connection():
    """Verify connection to Ollama service"""
    print_section("Verifying Ollama Connection")
    client = OllamaClient()
    print(f"Connecting to Ollama at: {client.base_url}")

    try:
        # Check if model is available
        is_available = await client.check_model_availability(MODEL_NAME)
        if not is_available:
            print(f"Warning: Model {MODEL_NAME} may not be available!")
            return False

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


async def test_ollama_client():
    """Test Ollama client functionality"""
    print_section("Testing Ollama Client")
    client = OllamaClient()

    try:
        # Test with different prompts
        test_cases = [
            ("What is AI?", None),
            ("Explain Python", "You are a programming instructor."),
        ]

        for prompt, system_prompt in test_cases:
            print(f"\nTesting prompt: {prompt}")
            if system_prompt:
                print(f"System prompt: {system_prompt}")

            response = await client.generate(
                prompt=prompt,
                model=MODEL_NAME,
                system_prompt=system_prompt,
                max_tokens=100,
            )

            print(f"Response preview: {response[:100]}...")
            assert len(response.strip()) > 0, "Response should not be empty"

        print("✓ Client tests passed")
        return True

    except Exception as e:
        print(f"✗ Client test failed: {str(e)}")
        return False


def test_sync_generate():
    """Test synchronous generation"""
    print_section("Testing Sync Generation")
    try:
        response = sync_generate(
            prompt="What is a computer?", model=MODEL_NAME, max_tokens=50
        )
        print(f"Response preview: {response[:100]}...")
        assert len(response.strip()) > 0, "Response should not be empty"

        print("✓ Sync generation test passed")
        return True

    except Exception as e:
        print(f"✗ Sync generation test failed: {str(e)}")
        return False


async def main():
    """Run complete test suite"""
    print_section("Starting Test Suite")

    # First verify connection
    if not await verify_ollama_connection():
        print("Failed to connect to Ollama service. Please check if it's running.")
        return False

    # Run tests
    results = {
        "ollama_client": await test_ollama_client(),
        "sync_generate": test_sync_generate(),
    }

    # Print summary
    print_section("Test Summary")
    for test_name, passed in results.items():
        print(f"{test_name}: {'✓ Passed' if passed else '✗ Failed'}")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
