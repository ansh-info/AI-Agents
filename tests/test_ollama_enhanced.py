import asyncio
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from clients.ollama_enhanced import EnhancedOllamaClient
from clients.semantic_scholar_client import PaperMetadata


def print_section(name: str):
    """Print a section header"""
    print(f"\n{'='*20} {name} {'='*20}")


@pytest.mark.asyncio
async def test_query_enhancement():
    """Test query enhancement functionality"""
    print_section("Testing Query Enhancement")

    client = EnhancedOllamaClient()
    test_queries = [
        "machine learning",
        "quantum computing applications",
        "neural networks in biology",
    ]

    for query in test_queries:
        enhanced = await client.enhance_search_query(query)
        print(f"Original: {query}")
        print(f"Enhanced: {enhanced}\n")

        assert isinstance(enhanced, str)
        assert len(enhanced) > 0


@pytest.mark.asyncio
async def test_result_summarization():
    """Test search result summarization"""
    print_section("Testing Result Summarization")

    client = EnhancedOllamaClient()

    # Create sample papers
    papers = [
        PaperMetadata(
            paperId="1",
            title="Deep Learning Advances",
            year=2023,
            authors=[{"name": "John Doe"}],
            abstract="Recent advances in deep learning...",
            citationCount=100,
        ),
        PaperMetadata(
            paperId="2",
            title="Neural Networks Today",
            year=2023,
            authors=[{"name": "Jane Smith"}],
            abstract="Modern neural network architectures...",
            citationCount=50,
        ),
    ]

    summary = await client.summarize_results(papers)
    print("Summary:", summary)

    assert isinstance(summary, str)
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_paper_analysis():
    """Test single paper analysis"""
    print_section("Testing Paper Analysis")

    client = EnhancedOllamaClient()

    paper = PaperMetadata(
        paperId="1",
        title="Transformer Architecture Innovations",
        year=2023,
        authors=[{"name": "Alice Johnson"}, {"name": "Bob Wilson"}],
        abstract="This paper presents novel improvements to transformer architectures...",
        citationCount=200,
    )

    analysis = await client.analyze_paper(paper)
    print("Analysis:", analysis)

    assert isinstance(analysis, str)
    assert len(analysis) > 0


@pytest.mark.asyncio
async def test_query_suggestions():
    """Test related query suggestions"""
    print_section("Testing Query Suggestions")

    client = EnhancedOllamaClient()
    query = "deep learning in computer vision"

    suggestions = await client.suggest_related_queries(query)
    print(f"Original query: {query}")
    print("Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


async def main():
    """Run all tests"""
    print("Starting Enhanced Ollama Integration Tests")

    await test_query_enhancement()
    await test_result_summarization()
    await test_paper_analysis()
    await test_query_suggestions()

    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
