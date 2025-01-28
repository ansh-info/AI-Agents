import asyncio
import json
from typing import Any, Dict

from agent.main_agent import MainAgent


async def test_intent_analysis(main_agent: MainAgent, query: str) -> Dict[str, Any]:
    """Test the intent analysis of a single query"""
    print(f"\n🔍 Testing intent analysis for: {query}")
    try:
        intent_result = await main_agent._determine_intent(query)
        print("Intent Analysis Result:")
        print(json.dumps(intent_result, indent=2))
        return intent_result
    except Exception as e:
        print(f"❌ Error in intent analysis: {str(e)}")
        return {}


async def compare_routing_approaches():
    """Compare keyword-based vs LLM-based routing"""
    main_agent = MainAgent()

    test_queries = [
        {
            "query": "Find papers about machine learning",
            "description": "Basic search query",
        },
        {
            "query": "Can you explain what transformer models are?",
            "description": "Conversational query",
        },
        {
            "query": "I need papers published after 2020 about deep learning with at least 100 citations",
            "description": "Complex search query with parameters",
        },
        {
            "query": "What were the results from my last search?",
            "description": "Context-dependent query",
        },
        {
            "query": "Show me papers by Geoffrey Hinton",
            "description": "Author-specific search",
        },
    ]

    print("\n=== Testing Routing Approaches ===")

    for test in test_queries:
        print(f"\n📝 Test Case: {test['description']}")
        print(f"Query: {test['query']}")

        # Test intent analysis
        intent_result = await test_intent_analysis(main_agent, test["query"])

        # Process full request
        print("\n🔄 Processing full request...")
        try:
            response = await main_agent.process_request(test["query"])

            print("\nResults:")
            print(f"Status: {response.status.value}")
            if response.error_message:
                print(f"Error: {response.error_message}")

            # Show last system response
            if response.memory.messages:
                last_message = response.memory.messages[-1]
                print(f"\nSystem Response: {last_message['content'][:200]}...")

            print("\nState Information:")
            print(f"Current Step: {response.current_step}")
            print(
                f"Search Context Active: {'results' in response.search_context.__dict__}"
            )

        except Exception as e:
            print(f"❌ Error processing request: {str(e)}")

        print("\n" + "=" * 50)


async def test_complex_scenarios():
    """Test more complex scenarios that should demonstrate LLM understanding"""
    main_agent = MainAgent()

    complex_queries = [
        "I'm interested in papers that bridge machine learning and cognitive science",
        "Find papers that challenge the current thinking in deep learning",
        "What are the most influential papers in AI safety from the past two years?",
        "Show me papers that connect reinforcement learning with neuroscience",
        "I need papers discussing limitations of large language models",
    ]

    print("\n=== Testing Complex Understanding ===")

    for query in complex_queries:
        print(f"\n🧪 Testing Complex Query: {query}")
        intent_result = await test_intent_analysis(main_agent, query)

        # Analyze the sophistication of the intent analysis
        if intent_result:
            print("\nIntent Analysis Quality Metrics:")
            print(f"- Parameter Extraction: {'search_params' in intent_result}")
            print(f"- Context Awareness: {'requires_context' in intent_result}")
            print(f"- Explanation Provided: {'explanation' in intent_result}")

            if "search_params" in intent_result:
                params = intent_result["search_params"]
                print(f"- Search Parameters Identified: {list(params.keys())}")

        print("\n" + "=" * 50)


async def main():
    print("Starting Routing Tests...")

    print("\n1️⃣ Testing Basic Routing")
    await compare_routing_approaches()

    print("\n2️⃣ Testing Complex Scenarios")
    await test_complex_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
