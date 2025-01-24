import asyncio
import json

from agent.enhanced_workflow import EnhancedWorkflowManager
from state.agent_state import AgentState


async def print_result(response: AgentState, test_name: str):
    """Helper function to print test results"""
    print(f"\n=== Test: {test_name} ===")
    print(f"Status: {response.status}")
    if response.error_message:
        print(f"Error: {response.error_message}")

    if response.memory and response.memory.messages:
        last_message = response.memory.messages[-1]
        print(f"Response: {last_message['content'][:200]}...")
    print("=" * 50)


async def test_agent():
    """Test all agent functionality"""
    # Initialize managers
    enhanced_workflow = EnhancedWorkflowManager()

    async def run_test(message: str, test_name: str):
        """Run a single test case"""
        print(f"\nExecuting test: {test_name}")
        try:
            response = await enhanced_workflow.process_command_async(message)
            await print_result(response, test_name)
            return response
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")
            return None

    # Test 1: Basic Conversation
    await run_test("Hi there! Can you help me with research?", "Basic Conversation")

    # Test 2: Paper Search
    search_result = await run_test(
        "Find recent papers about large language models published in the last 2 years",
        "Paper Search",
    )

    # Test 3: Paper Analysis
    if search_result and search_result.search_context.results:
        # Get first paper from previous search
        paper_num = 1
        await run_test(
            f"Can you analyze paper {paper_num} in detail?", "Paper Analysis"
        )

        # Test 4: Specific Question
        await run_test(
            f"What methodology did paper {paper_num} use?", "Specific Paper Question"
        )

        # Test 5: Paper Comparison
        await run_test(
            "Compare papers 1 and 2 from the search results", "Paper Comparison"
        )

    # Test 6: Invalid Paper Reference
    await run_test("Tell me about paper 999", "Invalid Paper Reference")

    # Test 7: System Health Check
    print("\n=== Testing System Health ===")
    health_status = await enhanced_workflow.check_workflow_health()
    print("Health Status:")
    print(json.dumps(health_status, indent=2))

    # Test 8: Complex Query
    await run_test(
        "Find papers about transformers in machine learning, focus on those with high citations",
        "Complex Search Query",
    )

    # Test 9: Error Handling
    await run_test(
        "",  # Empty query to test error handling
        "Error Handling - Empty Query",
    )

    # Test 10: State Management
    state_test_result = await run_test(
        "What were the papers we just found about transformers?",
        "State Management - Context Retention",
    )


async def test_tools():
    """Test individual tools"""
    workflow = EnhancedWorkflowManager()

    print("\n=== Testing Individual Tools ===")

    # Test Semantic Scholar Tool
    print("\nTesting Semantic Scholar Tool...")
    try:
        state = AgentState()
        state.add_message("user", "Find papers about neural networks")
        response = await workflow.semantic_scholar_tool._arun("neural networks")
        print(f"Search Results: {response[:200]}...")
    except Exception as e:
        print(f"Semantic Scholar Tool Error: {str(e)}")

    # Test Paper Analyzer Tool
    print("\nTesting Paper Analyzer Tool...")
    try:
        if state.search_context.results:
            paper = state.search_context.results[0]
            response = await workflow.paper_analyzer_tool._arun(
                paper_id=paper.paper_id, analysis_type="summary"
            )
            print(f"Analysis Results: {response[:200]}...")
    except Exception as e:
        print(f"Paper Analyzer Tool Error: {str(e)}")

    # Test Ollama Tool
    print("\nTesting Ollama Tool...")
    try:
        response = await workflow.ollama_tool._arun("Explain what a neural network is")
        print(f"Ollama Response: {response[:200]}...")
    except Exception as e:
        print(f"Ollama Tool Error: {str(e)}")


async def main():
    """Main test function"""
    print("Starting Agent Tests...")
    await test_agent()
    print("\nStarting Tool Tests...")
    await test_tools()
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
