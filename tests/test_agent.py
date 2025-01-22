# test_agent.py
import asyncio

from agent.workflow_manager import ResearchWorkflowManager


async def test_agent():
    # Initialize workflow manager
    workflow = ResearchWorkflowManager()

    async def process_and_print(message: str):
        print(f"\nTesting message: {message}")
        response = await workflow.process_message(message)
        if response and response.memory and response.memory.messages:
            # Get the last assistant message
            assistant_messages = [
                msg for msg in response.memory.messages if msg["role"] == "assistant"
            ]
            if assistant_messages:
                print(f"Response: {assistant_messages[-1]['content']}")
            else:
                print("No assistant response found")
        else:
            print("No response received")

    # Test conversation
    await process_and_print("Hi there!")

    # Test search
    await process_and_print("Find papers about large language models")


if __name__ == "__main__":
    asyncio.run(test_agent())
