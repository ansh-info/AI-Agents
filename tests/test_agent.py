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

            # Get all system/assistant messages
            system_messages = [
                msg for msg in messages if msg.get("role") in ["system", "assistant"]
            ]

            if system_messages:
                latest_response = system_messages[-1]
                print(f"Response: {latest_response['content']}")
                print(f"[DEBUG] Message role: {latest_response['role']}")
            else:
                print("No system response found")
                print("[DEBUG] Available messages:")
                for msg in messages:
                    print(
                        f"  - Role: {msg.get('role')}, Content: {msg.get('content')[:50]}..."
                    )
        else:
            print("No response received")
            if response:
                print(f"[DEBUG] State details: {response}")

    # Test conversation
    await process_and_print("Hi there!")

    # Test search
    await process_and_print("Find papers about large language models")


if __name__ == "__main__":
    asyncio.run(test_agent())
