import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from state.agent_state import AgentState, AgentStatus


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


async def test_hierarchical_workflow():
    """Test the hierarchical workflow implementation"""
    workflow = EnhancedWorkflowManager()

    # Test cases to verify different aspects of the system
    test_cases = [
        {
            "name": "Basic Search",
            "input": "Find recent papers about large language models",
            "expected_intent": "search",
        },
        {
            "name": "Paper Analysis",
            "input": "What is the methodology used in paper 1?",
            "expected_intent": "analyze",
        },
        {
            "name": "General Conversation",
            "input": "Can you explain what transformer models are?",
            "expected_intent": "conversation",
        },
        {
            "name": "Complex Query",
            "input": "Find papers about transformers and summarize their main findings",
            "expected_intent": "search",
        },
    ]

    for test in test_cases:
        print(f"\nExecuting test: {test['name']}")
        try:
            # Process request
            response = await workflow.process_command_async(test["input"])

            # Print results
            await print_result(response, test["name"])

            # Verify state updates
            assert response.status in [AgentStatus.SUCCESS, AgentStatus.ERROR]
            assert response.current_step is not None
            if response.status == AgentStatus.SUCCESS:
                assert (
                    len(response.memory.messages) >= 2
                )  # At least user input and system response

            print(f"✅ {test['name']} completed successfully")

        except Exception as e:
            print(f"❌ Error in {test['name']}: {str(e)}")


async def test_error_handling():
    """Test error handling in the workflow"""
    workflow = EnhancedWorkflowManager()

    error_test_cases = [
        {"name": "Empty Query", "input": "", "expected_error": True},
        {
            "name": "Invalid Paper Reference",
            "input": "Analyze paper 999",
            "expected_error": True,
        },
    ]

    for test in error_test_cases:
        print(f"\nExecuting error test: {test['name']}")
        try:
            response = await workflow.process_command_async(test["input"])
            assert (
                response.status == AgentStatus.ERROR
                if test["expected_error"]
                else AgentStatus.SUCCESS
            )
            print(f"✅ {test['name']} error handling working as expected")
        except Exception as e:
            print(f"❌ Error in {test['name']}: {str(e)}")


async def test_state_persistence():
    """Test state persistence across multiple interactions"""
    workflow = EnhancedWorkflowManager()

    # Sequence of interactions to test context maintenance
    interactions = [
        "Find papers about neural networks",
        "What is the methodology in paper 1?",
        "Compare papers 1 and 2",
        "What were the papers we found earlier?",
    ]

    print("\nTesting state persistence across interactions")
    for i, query in enumerate(interactions, 1):
        try:
            response = await workflow.process_command_async(query)
            print(f"\nInteraction {i}:")
            print(f"Query: {query}")
            print(
                f"State maintained: {len(response.memory.messages)} messages in history"
            )
            print(
                f"Search results tracked: {len(response.search_context.results) if response.search_context else 0} papers"
            )
        except Exception as e:
            print(f"❌ Error in interaction {i}: {str(e)}")


async def test_llm_routing():
    """Test the LLM-based intent routing and search"""
    print("\nTesting LLM-based routing and search...")
    workflow = EnhancedWorkflowManager()

    # Test cases focusing on intent analysis and routing
    test_cases = [
        {
            "name": "Specific Search Query",
            "input": "Find papers about transformers in deep learning from 2020 with at least 100 citations",
            "expected_components": ["intent_analysis", "semantic_scholar"],
        },
        {
            "name": "Ambiguous Query",
            "input": "What do you know about deep learning?",
            "expected_components": ["intent_analysis"],
        },
        {
            "name": "Complex Search Query",
            "input": "Find recent papers by Geoffrey Hinton about neural networks",
            "expected_components": ["intent_analysis", "semantic_scholar"],
        },
    ]

    for test in test_cases:
        print(f"\nExecuting test: {test['name']}")
        try:
            # Process request
            response = await workflow.process_command_async(test["input"])

            # Print debug information
            print(f"Test: {test['name']}")
            print(f"Input: {test['input']}")
            print("Response Messages:")
            for msg in response.memory.messages[
                -2:
            ]:  # Last user message and system response
                print(f"{msg['role']}: {msg['content'][:200]}...")

            # Verify components were called
            state_history = (
                response.state_history if hasattr(response, "state_history") else []
            )
            for component in test["expected_components"]:
                component_called = any(
                    component in str(state) for state in state_history
                )
                print(f"Component '{component}' called: {component_called}")

            print(f"✅ {test['name']} completed")

        except Exception as e:
            print(f"❌ Error in {test['name']}: {str(e)}")


# Add this to the main() function in test_agent.py
async def main():
    """Run all tests"""
    print("Starting Tests...")

    print("\n1. Testing Hierarchical Workflow")
    await test_hierarchical_workflow()

    print("\n2. Testing Error Handling")
    await test_error_handling()

    print("\n3. Testing State Persistence")
    await test_state_persistence()

    print("\n4. Testing LLM Routing")
    await test_llm_routing()

    print("\n5. Testing System Health")
    workflow = EnhancedWorkflowManager()
    health_status = await workflow.check_workflow_health()
    print("System Health Status:")
    print(json.dumps(health_status, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
