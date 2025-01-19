from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.semantic_scholar_client import (PaperMetadata, SearchFilters,
                                             SemanticScholarClient)


class SearchInput(BaseModel):
    """Schema for search parameters"""

    query: str = Field(..., description="The search query for finding papers")
    year_start: Optional[int] = Field(None, description="Start year filter")
    year_end: Optional[int] = Field(None, description="End year filter")
    min_citations: Optional[int] = Field(None, description="Minimum citations filter")
    max_results: Optional[int] = Field(
        10, description="Maximum number of results to return"
    )


class SemanticScholarTool(BaseTool):
    """Tool for Semantic Scholar paper search and retrieval"""

    name: str = "semantic_scholar_tool"
    description: str = """Use this tool to search for academic papers and retrieve paper information.
    
    Capabilities:
    1. Search for papers by topic, author, or title
    2. Filter by year, citation count
    3. Retrieve detailed paper information
    
    Input should be a natural language query describing what papers you're looking for.
    Example: "Find recent papers about large language models from the last 2 years"
    """
    args_schema: Type[BaseModel] = SearchInput

    # Private client instance
    _client: SemanticScholarClient = PrivateAttr()

    def __init__(self):
        """Initialize the tool with a Semantic Scholar client"""
        super().__init__()
        self._client = SemanticScholarClient()

    async def _arun(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
        max_results: int = 10,
    ) -> str:
        """Run the tool asynchronously"""
        try:
            # Create search filters
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            # Perform search
            results = await self._client.search_papers(
                query=query, filters=filters, limit=max_results
            )

            # Format results
            return self._format_results(results)

        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        """Synchronous run is not supported"""
        raise NotImplementedError("This tool only supports async execution")

    def _format_results(self, results) -> str:
        """Format search results into a clear response"""
        if not results.papers:
            return "No papers found matching your criteria."

        formatted_parts = [
            f"Found {results.total} papers. Here are the top {len(results.papers)} most relevant:\n"
        ]

        for i, paper in enumerate(results.papers, 1):
            paper_details = [
                f"\n{i}. {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}",
            ]

            if paper.abstract:
                paper_details.append(
                    f"Abstract: {paper.abstract[:300]}..."
                    if len(paper.abstract) > 300
                    else f"Abstract: {paper.abstract}"
                )

            paper_details.append(f"URL: {paper.url or 'Not available'}\n")
            formatted_parts.extend(paper_details)

        return "\n".join(formatted_parts)

    async def get_paper_details(self, paper_id: str) -> str:
        """Get detailed information about a specific paper"""
        try:
            paper = await self._client.get_paper_details(paper_id)

            details = [
                f"Title: {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'}",
                f"Citations: {paper.citations or 0}",
                f"Abstract: {paper.abstract or 'No abstract available'}",
                f"URL: {paper.url or 'Not available'}",
            ]

            return "\n".join(details)

        except Exception as e:
            return f"Error retrieving paper details: {str(e)}"

    async def check_health(self) -> bool:
        """Check if the tool is functioning properly"""
        try:
            # Test the API with a simple search
            await self._client.check_api_status()
            return True
        except:
            return False
