#!/usr/bin/env python3
import logging
import os
import sys
import uuid
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from agents.main_agent import get_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


def run_test_case(
    app, test_input: str, expected_agent: str, expected_tool: str, thread_id: str
) -> Dict:
    """
    Run a single test case through the agent workflow.

    Args:
        app: The compiled workflow app
        test_input (str): The user query to test
        expected_agent (str): Expected agent to handle the query
        expected_tool (str): Expected tool to be used
        thread_id (str): Thread ID for the MemorySaver

    Returns:
        Dict: The result from the workflow
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Testing input: {test_input}")
    logger.info(f"Expected agent: {expected_agent}")
    logger.info(f"Expected tool: {expected_tool}")

    # Prepare initial state with all required fields
    initial_state = {
        "messages": [HumanMessage(content=test_input)],
        "papers": [],
        "search_table": "",
        "next": None,
        "current_agent": None,
        "is_last_step": False,
    }

    try:
        # Run through workflow with configuration
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": str(uuid.uuid4()),
                "checkpoint_ns": "test",
            }
        }

        # Run through workflow
        result = app.invoke(initial_state, config=config)

        # Log results
        logger.info("Test completed")
        logger.info(f"Result: {result}")

        # Validate routing
        if result.get("current_agent") == expected_agent:
            logger.info(f"✓ Correctly routed to {expected_agent}")
        else:
            logger.info(
                f"✗ Expected {expected_agent}, but got {result.get('current_agent')}"
            )

        # Validate response content
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if expected_tool.lower() in str(last_message).lower():
                logger.info(f"✓ Used expected tool: {expected_tool}")
            else:
                logger.info(f"✗ Expected tool {expected_tool} not found in response")

        # Basic validation
        if result.get("messages"):
            logger.info("✓ Messages present in result")
        if result.get("papers"):
            logger.info("✓ Papers present in result")

        return result

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return {"error": str(e)}


def validate_test_result(result: Dict, test_case: Dict) -> bool:
    """
    Validate test results against expectations.

    Args:
        result: The test result
        test_case: The original test case with expectations

    Returns:
        bool: Whether the test passed
    """
    if result.get("error"):
        return False

    # Check if we got any response
    if not result.get("messages"):
        return False

    # Validate agent routing
    if result.get("current_agent") != test_case["expected_agent"]:
        return False

    # Validate tool usage (basic check in message content)
    last_message = result["messages"][-1]
    if test_case["expected_tool"].lower() not in str(last_message).lower():
        return False

    return True


def run_all_tests():
    """Run all test cases"""
    logger.info("Starting hierarchical agent tests")

    # Generate a unique thread ID for this test run
    thread_id = str(uuid.uuid4())

    # Initialize app
    app = get_app(thread_id)

    # Test cases - only for s2_agent as it's currently implemented
    test_cases = [
        {
            "input": "Find me papers on machine learning",
            "expected_agent": "s2_agent",
            "expected_tool": "search_tool",
        },
        {
            "input": "Find papers similar to paper ID: 649def34f8be52c8b66281af98ae884c09aef38b",
            "expected_agent": "s2_agent",
            "expected_tool": "single_paper_recommendations",
        },
        {
            "input": "Find papers related to both papers: 649def34f8be52c8b66281af98ae884c09aef38b and 7d7935bce46753c5e2868d8c268f1e6ff3d45396",
            "expected_agent": "s2_agent",
            "expected_tool": "multi_paper_recommendations",
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nRunning test case {i}")
        result = run_test_case(
            app,
            test_case["input"],
            test_case["expected_agent"],
            test_case["expected_tool"],
            thread_id,
        )
        test_passed = validate_test_result(result, test_case)
        results.append({"case": test_case, "result": result, "passed": test_passed})

    # Print summary
    logger.info("\nTest Summary:")
    total_passed = sum(1 for r in results if r["passed"])
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Passed: {total_passed}")
    logger.info(f"Failed: {len(results) - total_passed}")

    for i, result in enumerate(results, 1):
        status = "✓ Passed" if result["passed"] else "✗ Failed"
        logger.info(f"Test {i}: {status}")
        if not result["passed"]:
            logger.info(f"  Input: {result['case']['input']}")
            if result["result"].get("error"):
                logger.info(f"  Error: {result['result']['error']}")


if __name__ == "__main__":
    run_all_tests()
