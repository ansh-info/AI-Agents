from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.semantic_scholar_client import SearchFilters, SemanticScholarClient
from state.agent_state import AgentState


class SearchInput(BaseModel):
    """Enhanced schema for search parameters"""

    query: str = Field(..., description="The search query for finding papers")
    year_start: Optional[int] = Field(None, description="Start year filter")
    year_end: Optional[int] = Field(None, description="End year filter")
    min_citations: Optional[int] = Field(None, description="Minimum citations filter")
    max_results: Optional[int] = Field(
        10, description="Maximum number of results to return"
    )


class SemanticScholarTool(BaseTool):
    """Enhanced tool for Semantic Scholar paper search and retrieval"""

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

    # Private client instance using PrivateAttr for non-serializable fields
    _client: SemanticScholarClient = PrivateAttr()
    _state: Optional[AgentState] = PrivateAttr()

    def __init__(self, state: Optional[AgentState] = None):
        """Initialize the tool with a Semantic Scholar client and optional state"""
        super().__init__()
        self._client = SemanticScholarClient()
        self._state = state
        print("[DEBUG] Initialized SemanticScholarTool")

    async def _arun(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
        max_results: int = 10,
    ) -> str:
        """Run the tool asynchronously with enhanced error handling and state management"""
        try:
            print(f"[DEBUG] SemanticScholarTool: Processing query: {query}")

            # Create search filters
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            # Perform search
            results = await self._client.search_papers(
                query=query, filters=filters, limit=max_results
            )

            # Update state if available
            if self._state and results.papers:
                for paper in results.papers:
                    self._state.search_context.add_paper(
                        {
                            "paperId": paper.paperId,
                            "title": paper.title,
                            "abstract": paper.abstract,
                            "year": paper.year,
                            "authors": [
                                {"name": a.name, "authorId": a.authorId}
                                for a in paper.authors
                            ],
                            "citations": paper.citations,
                            "url": paper.url,
                        }
                    )

            # Format and return results
            return self._format_results(results)

        except Exception as e:
            error_msg = f"Error performing search: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    def _run(self, query: str, **kwargs) -> str:
        """Synchronous run is not supported"""
        raise NotImplementedError("This tool only supports async execution")

    def _format_results(self, results) -> str:
        """Format search results with enhanced structure"""
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

    def set_state(self, state: AgentState):
        """Set or update the current state"""
        self._state = state
        print("[DEBUG] Updated SemanticScholarTool state")

    async def check_health(self) -> bool:
        """Check if the tool is functioning properly"""
        try:
            return await self._client.check_api_status()
        except Exception as e:
            print(f"[DEBUG] Health check failed: {str(e)}")
            return False
