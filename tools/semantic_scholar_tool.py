from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.semantic_scholar_client import (
    SearchFilters,
    SemanticScholarClient,
)


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
        """Initialize the tool"""
        super().__init__()
        print("[DEBUG] Initializing SemanticScholarTool")
        self._client = SemanticScholarClient()
        print("[DEBUG] SemanticScholarTool initialized with client")

    async def _arun(self, query: str, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            print(f"[DEBUG] SemanticScholarTool: Starting search with query: {query}")

            # Add delay between requests
            await asyncio.sleep(1)  # Add 1 second delay

            # Create search filters
            filters = SearchFilters(**kwargs)

            # Perform search with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results = await self._client.search_papers(
                        query=query, filters=filters, limit=10
                    )
                    return self._format_results(results)
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(f"[DEBUG] Rate limited, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise

        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        """Synchronous run is not supported"""
        raise NotImplementedError("This tool only supports async execution")

    def _format_results(self, results) -> str:
        """Format search results with logging"""
        try:
            if not results.papers:
                print("[DEBUG] No papers found in results")
                return "No papers found matching your criteria."

            print(f"[DEBUG] Formatting {len(results.papers)} papers")
            formatted_parts = [
                f"Found {results.total} papers. Here are the most relevant ones:\n"
            ]

            for i, paper in enumerate(results.papers, 1):
                print(f"[DEBUG] Formatting paper {i}: {paper.title[:50]}...")
                # Rest of your formatting code...

            response = "\n".join(formatted_parts)
            print(f"[DEBUG] Formatted response created, length: {len(response)}")
            return response

        except Exception as e:
            error_msg = f"Error formatting results: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

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
