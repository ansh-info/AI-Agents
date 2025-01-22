# test_agent.py
import asyncio

from agent.workflow_manager import ResearchWorkflowManager


async def test_agent():
    # Initialize workflow manager
    workflow = ResearchWorkflowManager()

    # Test conversation
    print("\nTesting conversation...")
    response = await workflow.process_message("Hi there!")
    if response.memory.messages:
        last_message = response.memory.messages[-1]
        print(f"Response: {last_message['content']}")
    else:
        print("No response received")

    # Test paper search
    print("\nTesting paper search...")
    response = await workflow.process_message("Find papers about large language models")
    if response.memory.messages:
        last_message = response.memory.messages[-1]
        print(f"Response: {last_message['content']}")
    else:
        print("No response received")


if __name__ == "__main__":
    asyncio.run(test_agent())
