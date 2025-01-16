from typing import Annotated, Any, Dict, List, Optional

from langchain_core.tools import tool
from semantic_scholar_tool import SemanticScholarTool


class ResearchTools:
    """Collection of research-related tools for the agent."""

    def __init__(self):
        self.s2_tool = SemanticScholarTool()

    @tool
    async def search_papers(
        self,
        query: Annotated[str, "Search query for academic papers"],
        year_start: Annotated[Optional[int], "Start year filter"] = None,
        year_end: Annotated[Optional[int], "End year filter"] = None,
        min_citations: Annotated[Optional[int], "Minimum citations filter"] = None,
    ) -> str:
        """Search for academic papers using Semantic Scholar."""
        return await self.s2_tool.search_papers(
            query=query,
            year_start=year_start,
            year_end=year_end,
            min_citations=min_citations,
        )

    @tool
    async def get_paper_details(
        self, paper_id: Annotated[str, "The ID of the paper to retrieve details for"]
    ) -> str:
        """Get detailed information about a specific paper."""
        return await self.s2_tool.get_paper_details(paper_id)

    @tool
    async def advanced_search(
        self,
        filters: Annotated[Dict[str, Any], "Advanced search filters"],
        limit: Annotated[int, "Maximum number of results"] = 10,
    ) -> str:
        """Perform advanced search with custom filters."""
        return await self.s2_tool.search_by_filters(filters, limit)
