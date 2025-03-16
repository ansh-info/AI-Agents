#!/usr/bin/env python3
import logging
import sys
import uuid
from typing import Dict, List

from dotenv import load_dotenv

from agents.main_agent import get_app
from state.shared_state import Talk2Papers  # Import state class

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
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing input: {test_input}")
    logger.info(f"Expected agent: {expected_agent}")
    logger.info(f"Expected tool: {expected_tool}")

    # Prepare initial state with all required fields
    initial_state = {
        "messages": [{"role": "user", "content": test_input}],
        "papers": [],
        "search_table": "",
        "next": None,
        "current_agent": None,
        "is_last_step": False,  # Initialize is_last_step
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

        # Basic validation
        if result.get("messages"):
            logger.info("✓ Messages present in result")
        if result.get("papers"):
            logger.info("✓ Papers present in result")

        return result

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return {"error": str(e)}


def run_all_tests():
    """Run all test cases"""
    logger.info("Starting hierarchical agent tests")

    # Generate a unique thread ID for this test run
    thread_id = str(uuid.uuid4())

    # Initialize app
    app = get_app(thread_id)

    # Test cases
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
    for test_case in test_cases:
        result = run_test_case(
            app,
            test_case["input"],
            test_case["expected_agent"],
            test_case["expected_tool"],
            thread_id,
        )
        results.append(result)

    logger.info("\nTest Summary:")
    for i, result in enumerate(results):
        status = (
            "✓ Passed" if not result.get("error") else f"✗ Failed: {result['error']}"
        )
        logger.info(f"Test {i+1}: {status}")


if __name__ == "__main__":
    run_all_tests()
