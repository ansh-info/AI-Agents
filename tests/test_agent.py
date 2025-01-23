# test_agent.py
import asyncio

from agent.workflow_manager import ResearchWorkflowManager


async def test_agent():
    # Initialize workflow manager
    workflow = ResearchWorkflowManager()

    async def process_and_print(message: str):
        print(f"\nTesting message: {message}")
        response = await workflow.process_message(message)
        print(f"[DEBUG] Response state status: {response.status}")

        if response and response.memory and response.memory.messages:
            messages = response.memory.messages
            print(f"[DEBUG] Total messages: {len(messages)}")

            # Get the last system message
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            if system_messages:
                print(f"Response: {system_messages[-1]['content']}")
            else:
                print("No system response found")
        else:
            print("No response received")

    # Test conversation
    await process_and_print("Hi there!")

    # Test search
    await process_and_print("Find papers about large language models")


if __name__ == "__main__":
    asyncio.run(test_agent())
