# test_agent.py
import asyncio

from agent.workflow_manager import ResearchWorkflowManager


async def test_agent():
    # Initialize workflow manager
    workflow = ResearchWorkflowManager()

    # Test conversation
    print("\nTesting conversation...")
    response = await workflow.process_message("Hi there!")
    print(f"Response: {response.memory.messages[-1]['content']}")

    # Test paper search
    print("\nTesting paper search...")
    response = await workflow.process_message("Find papers about large language models")
    print(f"Response: {response.memory.messages[-1]['content']}")


if __name__ == "__main__":
    asyncio.run(test_agent())
