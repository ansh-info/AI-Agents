import asyncio
import os
import sys
from typing import Optional

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from clients.semantic_scholar_client import (PaperMetadata, SearchResults,
                                             SemanticScholarClient)


def print_section(name: str):
    """Print a section header"""
    print(f"\n{'='*20} {name} {'='*20}")


def print_paper_info(paper: PaperMetadata):
    """Helper to print paper information"""
    print(f"\n--- Paper Info ---")
    print(f"Title: {paper.title}")
    print(f"Year: {paper.year}")
    print(f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}")
    if paper.abstract:
        print(f"Abstract: {paper.abstract[:200]}...")
    print(f"Citations: {paper.citationCount}")
    print(f"References: {paper.referenceCount}")


class TestSemanticScholarClient:
    """Test suite for Semantic Scholar API client"""

    @pytest.fixture
    async def client(self):
        """Fixture to create client instance"""
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        return SemanticScholarClient(api_key=api_key)

    @pytest.mark.asyncio
    async def test_search_basic(self, client):
        """Test basic paper search functionality"""
        print_section("Testing Basic Search")

        results = await client.search_papers(query="large language models", limit=5)

        assert isinstance(results, SearchResults)
        assert len(results.papers) <= 5
        assert results.total > 0
        assert all(isinstance(p, PaperMetadata) for p in results.papers)

        print(
            f"Found {results.total} papers. Showing first {len(results.papers)} results:"
        )
        for paper in results.papers:
            print_paper_info(paper)

    @pytest.mark.asyncio
    async def test_search_with_year(self, client):
        """Test search with year filter"""
        print_section("Testing Search with Year Filter")

        results = await client.search_papers(
            query="transformers nlp", limit=3, year=2023
        )

        assert all(p.year == 2023 for p in results.papers)
        print(f"Found {len(results.papers)} papers from 2023:")
        for paper in results.papers:
            print_paper_info(paper)

    @pytest.mark.asyncio
    async def test_search_pagination(self, client):
        """Test search pagination"""
        print_section("Testing Pagination")

        # First page
        page1 = await client.search_papers(
            query="GPT language model", offset=0, limit=5
        )

        # Second page
        page2 = await client.search_papers(
            query="GPT language model", offset=5, limit=5
        )

        assert len(page1.papers) == 5
        assert page1.offset == 0
        assert page2.offset == 5

        # Check for unique papers
        page1_ids = {p.paperId for p in page1.papers}
        page2_ids = {p.paperId for p in page2.papers}
        assert not page1_ids.intersection(page2_ids)

        print("Pagination test successful:")
        print(f"Page 1: {len(page1.papers)} papers, offset {page1.offset}")
        print(f"Page 2: {len(page2.papers)} papers, offset {page2.offset}")

    @pytest.mark.asyncio
    async def test_get_paper_details(self, client):
        """Test retrieving detailed paper information"""
        print_section("Testing Paper Details")

        # First get a paper ID through search
        search_results = await client.search_papers(
            query="attention is all you need", limit=1
        )
        assert len(search_results.papers) > 0
        paper_id = search_results.papers[0].paperId

        # Get detailed information
        paper = await client.get_paper_details(paper_id)

        assert isinstance(paper, PaperMetadata)
        assert paper.paperId == paper_id
        assert paper.title
        assert paper.authors

        print_paper_info(paper)

    @pytest.mark.asyncio
    async def test_search_empty_results(self, client):
        """Test search with query that should return no results"""
        print_section("Testing Empty Results")

        results = await client.search_papers(
            query="thisisaverylongquerythatwillnotmatchanypapers12345", limit=5
        )

        assert isinstance(results, SearchResults)
        assert len(results.papers) == 0
        assert results.total == 0

        print("Empty results test successful")

    @pytest.mark.asyncio
    async def test_field_selection(self, client):
        """Test field selection in API response"""
        print_section("Testing Field Selection")

        # Request only specific fields
        fields = ["paperId", "title", "year"]
        results = await client.search_papers(
            query="machine learning", limit=1, fields=fields
        )

        paper = results.papers[0]
        assert paper.paperId
        assert paper.title
        assert paper.abstract is None  # Not requested

        print("Field selection test successful:")
        print(f"Requested fields: {fields}")
        print(f"Paper data: {paper.dict()}")

    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling with invalid inputs"""
        print_section("Testing Error Handling")

        # Test invalid paper ID
        with pytest.raises(Exception) as exc_info:
            await client.get_paper_details("invalid_paper_id_12345")
        assert "API error" in str(exc_info.value)

        # Test oversized limit
        results = await client.search_papers(
            query="AI", limit=150  # Should be capped at 100
        )
        assert len(results.papers) <= 100

        print("Error handling tests successful")


if __name__ == "__main__":
    print("Starting Semantic Scholar Client Tests")
    pytest.main([__file__, "-v"])
