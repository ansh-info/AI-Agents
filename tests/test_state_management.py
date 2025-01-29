import asyncio

from agent.workflow_graph import WorkflowGraph
from state.agent_state import AgentStatus


async def test_state_management():
    """Test the enhanced state management functionality"""
    workflow = WorkflowGraph()

    # Test sequence of interactions
    test_sequence = [
        {
            "query": "Find papers about large language models",
            "check": lambda state: (
                len(state.search_context.results) > 0,
                "Search should return results",
            ),
        },
        {
            "query": "What was my last search about?",
            "check": lambda state: (
                "language models" in state.memory.messages[-1]["content"].lower(),
                "Should reference previous search about language models",
            ),
        },
        {
            "query": "Tell me about the first paper",
            "check": lambda state: (
                state.memory.focused_paper is not None,
                "Should have a focused paper",
            ),
        },
        {
            "query": "What paper were we just discussing?",
            "check": lambda state: (
                (
                    state.memory.focused_paper
                    and state.memory.focused_paper.title
                    in state.memory.messages[-1]["content"]
                )
                or "no paper" in state.memory.messages[-1]["content"].lower(),
                "Should mention the previously discussed paper",
            ),
        },
    ]

    results = []
    for test in test_sequence:
        print(f"\n🧪 Testing: {test['query']}")

        # Process request
        state = await workflow.process_request(test["query"])

        # Run checks
        passed, message = test["check"](state)

        # Verify basic state properties
        basic_checks = [
            (state.status != AgentStatus.ERROR, "State should not be in error"),
            (len(state.memory.messages) > 0, "Should have messages in memory"),
            (state.last_update is not None, "Should have last_update timestamp"),
        ]

        # Combine all checks
        all_passed = passed and all(check[0] for check in basic_checks)

        # Format result
        result = {
            "query": test["query"],
            "passed": all_passed,
            "message": message,
            "state_status": state.status,
            "message_count": len(state.memory.messages),
            "last_message": state.memory.messages[-1]["content"][:100] + "...",
        }
        results.append(result)

        # Print result
        print(f"✓ Passed: {all_passed}")
        if not all_passed:
            print(f"✗ Failed: {message}")
            for check in basic_checks:
                if not check[0]:
                    print(f"✗ Failed: {check[1]}")

        print(f"Last message: {result['last_message']}")

    return results


async def test_conversation_history():
    """Test specific conversation history handling"""
    workflow = WorkflowGraph()

    # Test history-specific queries
    history_queries = [
        "Find papers about transformers",
        "What did we just search for?",
        "Find papers by Geoffrey Hinton",
        "What were the results of my previous search?",
        "What topics have we discussed so far?",
    ]

    results = []
    for query in history_queries:
        print(f"\n🧪 Testing history query: {query}")

        state = await workflow.process_request(query)

        # Verify history tracking
        result = {
            "query": query,
            "message_count": len(state.memory.messages),
            "has_history": len(state.memory.messages) > 1,
            "context_preserved": state.memory.current_context is not None,
            "last_response": state.memory.messages[-1]["content"][:100] + "...",
        }
        results.append(result)

        print(f"Messages in history: {result['message_count']}")
        print(f"Context preserved: {result['context_preserved']}")
        print(f"Last response: {result['last_response']}")

    return results


async def run_all_tests():
    """Run all workflow tests"""
    print("\n=== Running State Management Tests ===")
    state_results = await test_state_management()

    print("\n=== Running Conversation History Tests ===")
    history_results = await test_conversation_history()

    return {"state_management": state_results, "conversation_history": history_results}


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())

    # Print summary
    print("\n=== Test Summary ===")
    print("State Management Tests:")
    passed = sum(1 for r in results["state_management"] if r["passed"])
    total = len(results["state_management"])
    print(f"Passed: {passed}/{total} tests")

    print("\nConversation History Tests:")
    history_total = len(results["conversation_history"])
    print(f"Completed {history_total} history queries")

    # Print any failures
    failures = [r for r in results["state_management"] if not r["passed"]]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"- Query: {f['query']}")
            print(f"  Message: {f['message']}")
