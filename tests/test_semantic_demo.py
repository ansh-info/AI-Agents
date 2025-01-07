import asyncio
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.enhanced_workflow import EnhancedWorkflowManager


async def run_demo():
    # Initialize the workflow manager
    manager = EnhancedWorkflowManager()

    # Test different commands
    commands = ["help", "search large language models", "next", "prev"]

    for command in commands:
        print(f"\n>>> Executing command: {command}")
        state = await manager.process_command_async(command)

        print("\nResponse:")
        for message in state.memory.messages:
            if message["role"] == "system":
                print(message["content"])

        if state.search_context.query:
            print(f"\nCurrent page: {state.search_context.current_page}")
            print(f"Total results: {state.search_context.total_results}")


if __name__ == "__main__":
    print("Starting Paper Search Demo")
    print("=" * 50)
    asyncio.run(run_demo())
