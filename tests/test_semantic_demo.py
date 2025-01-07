import asyncio
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.enhanced_workflow import EnhancedWorkflowManager


async def run_demo():
    # Initialize the workflow manager
    manager = EnhancedWorkflowManager()

    commands = [
        "help",
        "search large language models",
        "next",  # Should show next page
        "next",  # Should show another page
        "prev",  # Should go back
        "clear",  # Clear the search
        "search transformers",  # New search
        "help",  # Show help again
    ]

    for command in commands:
        print(f"\n{'='*50}")
        print(f">>> Executing command: {command}")
        state = await manager.process_command_async(command)

        print("\nResponse:")
        for message in state.memory.messages[-1:]:  # Only show last message
            if message["role"] == "system":
                print(message["content"])

        if state.search_context.query:
            print(f"\nSearch Info:")
            print(f"Query: {state.search_context.query}")
            print(f"Current page: {state.search_context.current_page}")
            print(f"Total results: {state.search_context.total_results}")


if __name__ == "__main__":
    print("Starting Enhanced Paper Search Demo")
    print("=" * 50)
    asyncio.run(run_demo())
