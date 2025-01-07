import asyncio
import os
import sys
from typing import Optional

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus

# Configuration
MODEL_NAME = "llama3.2:1b"


def print_section(name: str):
    """Print a section header"""
    print(f"\n{'='*20} {name} {'='*20}")


def print_state_info(state: AgentState, title: str = "State Info"):
    """Helper to print state information"""
    print(f"\n--- {title} ---")
    print(f"Status: {state.status}")
    print(f"Current step: {state.current_step}")
    if state.error_message:
        print(f"Error: {state.error_message}")
    print("\nMessages:")
    for msg in state.memory.messages:
        print(f"- {msg['role']}: {msg['content']}")
    if state.search_context.query:
        print(f"\nSearch query: {state.search_context.query}")
        print(f"Page: {state.search_context.current_page}")
        print(f"Total results: {state.search_context.total_results}")


class TestIntegratedWorkflow:
    """Test suite for integrated workflow with Semantic Scholar"""

    @pytest.fixture
    async def manager_fixture(self):
        """Fixture to create workflow manager instance for pytest"""
        return EnhancedWorkflowManager(model_name=MODEL_NAME)

    async def create_manager(self):
        """Helper function to create workflow manager instance"""
        return EnhancedWorkflowManager(model_name=MODEL_NAME)

    @pytest.mark.asyncio
    async def test_semantic_scholar_connection(self):
        """Test Semantic Scholar API connection"""
        print_section("Testing Semantic Scholar Connection")

        client = SemanticScholarClient()
        results = await client.search_papers(query="test query", limit=1)

        assert results.total > 0, "Should return some results"
        assert len(results.papers) <= 1, "Should respect result limit"
        print("✓ Semantic Scholar connection test passed")

    @pytest.mark.asyncio
    async def test_basic_commands(self, manager_fixture):
        """Test basic command processing"""
        print_section("Testing Basic Commands")

        # Test help command
        state = await manager_fixture.process_command_async("help")
        assert state.status == AgentStatus.SUCCESS
        assert state.current_step == "help_displayed"
        assert any(
            "Available commands" in msg["content"] for msg in state.memory.messages
        )
        print("✓ Help command test passed")

        # Test invalid command
        state = await manager_fixture.process_command_async("invalid command")
        assert state.status == AgentStatus.SUCCESS
        assert state.current_step == "command_processed"
        assert any("Unknown command" in msg["content"] for msg in state.memory.messages)
        print("✓ Invalid command test passed")

    @pytest.mark.asyncio
    async def test_search_workflow(self, manager_fixture):
        """Test search command workflow"""
        print_section("Testing Search Workflow")

        # Test basic search
        query = "large language models"
        state = await manager_fixture.process_command_async(f"search {query}")

        assert state.status == AgentStatus.SUCCESS
        assert state.current_step == "search_completed"
        assert state.search_context.query == query
        assert state.search_context.total_results > 0
        assert len(state.search_context.results) <= manager_fixture.search_limit
        print("✓ Basic search test passed")

        # Test empty search command
        state = await manager_fixture.process_command_async("search")
        assert state.current_step == "invalid_search"
        assert any(
            "Please provide a search query" in msg["content"]
            for msg in state.memory.messages
        )
        print("✓ Empty search test passed")

    @pytest.mark.asyncio
    async def test_pagination(self, manager_fixture):
        """Test pagination functionality"""
        print_section("Testing Pagination")

        # First perform a search
        state = await manager_fixture.process_command_async("search machine learning")
        initial_page = state.search_context.current_page

        # Test next page
        if state.search_context.total_results > manager_fixture.search_limit:
            state = await manager_fixture.process_command_async("next")
            assert state.status == AgentStatus.SUCCESS
            assert state.search_context.current_page == initial_page + 1
            assert len(state.search_context.results) <= manager_fixture.search_limit
            print("✓ Next page test passed")

            # Test previous page
            state = await manager_fixture.process_command_async("prev")
            assert state.status == AgentStatus.SUCCESS
            assert state.search_context.current_page == initial_page
            print("✓ Previous page test passed")
        else:
            print("Skipping pagination tests (not enough results)")

    @pytest.mark.asyncio
    async def test_error_handling(self, manager_fixture):
        """Test error handling scenarios"""
        print_section("Testing Error Handling")

        # Test pagination without search
        manager_fixture.reset_state()
        state = await manager_fixture.process_command_async("next")
        assert "No active search" in state.memory.messages[-1]["content"]
        print("✓ Pagination without search test passed")

        # Test invalid search query
        state = await manager_fixture.process_command_async("search " + "a" * 1000)
        assert state.status == AgentStatus.SUCCESS or state.status == AgentStatus.ERROR
        print("✓ Invalid query test passed")

    @pytest.mark.asyncio
    async def test_result_formatting(self, manager_fixture):
        """Test search result formatting"""
        print_section("Testing Result Formatting")

        state = await manager_fixture.process_command_async("search neural networks")

        # Check result formatting
        results_msg = [msg for msg in state.memory.messages if msg["role"] == "system"][
            -1
        ]["content"]

        assert "Found" in results_msg
        assert "Authors:" in results_msg
        assert "Abstract:" in results_msg
        assert "Citations:" in results_msg
        print("✓ Result formatting test passed")

    @pytest.mark.asyncio
    async def test_state_management(self, manager_fixture):
        """Test state management across commands"""
        print_section("Testing State Management")

        # Perform search
        state = await manager_fixture.process_command_async("search deep learning")
        initial_query = state.search_context.query
        initial_results = state.search_context.total_results

        # Execute next page
        state = await manager_fixture.process_command_async("next")

        # Verify state persistence
        assert state.search_context.query == initial_query
        assert state.search_context.total_results == initial_results
        print("✓ State management test passed")


async def run_all_tests():
    """Run all tests sequentially"""
    try:
        workflow = TestIntegratedWorkflow()
        manager = await workflow.create_manager()

        print("\nStarting Integrated Workflow Tests")
        print("=" * 50)

        # Run all tests
        await workflow.test_semantic_scholar_connection()
        await workflow.test_basic_commands(manager)
        await workflow.test_search_workflow(manager)
        await workflow.test_pagination(manager)
        await workflow.test_error_handling(manager)
        await workflow.test_result_formatting(manager)
        await workflow.test_state_management(manager)

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"\nTest suite failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting Integrated Workflow Test Suite")
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
