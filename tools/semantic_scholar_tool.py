from typing import Any, Dict, Optional, Type

from langchain.tools import BaseTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from clients.semantic_scholar_client import SearchFilters, SemanticScholarClient
from state.agent_state import AgentState


class SearchInput(BaseModel):
    """Schema for paper search parameters"""

    query: str = Field(..., description="The search query to find relevant papers")
    year_start: Optional[int] = Field(
        None, description="Filter papers from this year onwards"
    )
    year_end: Optional[int] = Field(None, description="Filter papers until this year")
    min_citations: Optional[int] = Field(
        None, description="Minimum number of citations"
    )
    max_results: Optional[int] = Field(
        10, description="Maximum number of results to return"
    )


class SemanticScholarTool(BaseTool):
    """Tool for searching academic papers using Semantic Scholar API"""

    # Add type annotation for name and description
    name: str = "semantic_scholar_search"
    description: str = """Use this tool to search for academic papers and retrieve their information.
    
    This tool can:
    - Search for papers by topic, keywords, or authors
    - Filter results by year and citation count
    - Retrieve detailed paper information including abstracts
    - Return formatted results with paper details
    
    Input should be a search query describing what papers you want to find.
    """
    args_schema: Type[BaseModel] = SearchInput

    def __init__(self, state: Optional[AgentState] = None):
        """Initialize the tool with optional state"""
        super().__init__()
        self._client = SemanticScholarClient()
        self._state = state
        print("[DEBUG] Initialized SemanticScholarTool")

    def _run(self, query: str, **kwargs) -> str:
        """Synchronous execution not supported"""
        raise NotImplementedError("This tool only supports async execution")

    async def _arun(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Execute search with proper parameter handling"""
        try:
            print(f"[DEBUG] SemanticScholarTool: Processing query: {query}")

            # Clean query
            clean_query = " ".join(query.split())[:1000]  # Limit query length

            # Create search filters with proper formatting
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            # Get properly formatted parameters
            params = {
                "query": clean_query,
                "offset": 0,
                "limit": max_results,
                "fields": "paperId,title,abstract,year,authors,citationCount,url",
            }
            params.update(filters.to_params())

            print(f"[DEBUG] Making request with params: {params}")
            results = await self._client.search_papers(
                query=clean_query, filters=filters, limit=max_results
            )

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

            return {
                "status": "success",
                "total_results": results.total,
                "papers": [
                    {
                        "id": paper.paperId,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "year": paper.year,
                        "citations": paper.citations,
                        "abstract": (
                            paper.abstract[:300] + "..."
                            if paper.abstract
                            else "No abstract available"
                        ),
                        "url": paper.url,
                    }
                    for paper in results.papers
                ],
            }

        except Exception as e:
            print(f"[DEBUG] Error in search: {str(e)}")
            return {"status": "error", "error": str(e), "papers": []}

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
