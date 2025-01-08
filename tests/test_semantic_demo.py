import asyncio
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.enhanced_workflow import EnhancedWorkflowManager


async def run_demo():
    manager = EnhancedWorkflowManager()

    commands = [
        "help",
        "search large language models year:2023",  # Test year filter
        "next",  # Test pagination
        "search transformers citations>5000",  # Test citation filter
        "search gpt-4 sort:citations",  # Test sorting
        "clear",  # Test clear
        "help",  # Show help again
    ]

    for command in commands:
        print(f"\n{'='*50}")
        print(f">>> Executing command: {command}")
        state = await manager.process_command_async(command)

        print("\nResponse:")
        if state.error_message:
            print(f"Error: {state.error_message}")

        for message in state.memory.messages[-1:]:
            if message["role"] == "system":
                print(message["content"])

        if state.search_context.query:
            print(f"\nSearch Info:")
            print(f"Query: {state.search_context.query}")
            print(f"Current page: {state.search_context.current_page}")
            print(f"Total results: {state.search_context.total_results}")
            if state.search_context.current_filters:
                print("Applied filters:", state.search_context.current_filters)


if __name__ == "__main__":
    print("Starting Enhanced Paper Search Demo")
    print("=" * 50)
    asyncio.run(run_demo())
