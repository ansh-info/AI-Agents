from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, tool
from semantic_scholar_client import (PaperMetadata, SearchFilters,
                                     SemanticScholarClient)


class SemanticScholarTool(BaseTool):
    """Tool for interacting with Semantic Scholar API"""

    def __init__(self):
        self.client = SemanticScholarClient()
        super().__init__()

    @tool
    async def search_papers(
        self,
        query: Annotated[str, "The search query for finding papers"],
        year_start: Annotated[Optional[int], "Start year filter"] = None,
        year_end: Annotated[Optional[int], "End year filter"] = None,
        min_citations: Annotated[Optional[int], "Minimum citations"] = None,
    ) -> str:
        """Search for academic papers on Semantic Scholar."""
        try:
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            results = await self.client.search_papers(
                query=query, filters=filters, limit=10
            )

            # Format results for agent consumption
            response_parts = [
                f"Found {results.total} papers related to '{query}'. Here are the most relevant papers:\n"
            ]

            for i, paper in enumerate(results.papers, 1):
                paper_info = [
                    f"[{i}] {paper.title}",
                    f"Authors: {', '.join(a.name for a in paper.authors)}",
                    f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}",
                    f"Abstract: {paper.abstract[:300] + '...' if paper.abstract else 'No abstract available'}",
                    f"URL: {paper.url or 'No URL'}\n",
                ]
                response_parts.extend(paper_info)

            return "\n".join(response_parts)

        except Exception as e:
            return f"Error searching papers: {str(e)}"

    @tool
    async def get_paper_details(
        self, paper_id: Annotated[str, "ID of the paper to fetch details for"]
    ) -> str:
        """Get detailed information about a specific paper."""
        try:
            paper = await self.client.get_paper_details(paper_id)

            response = [
                f"Title: {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'}",
                f"Citations: {paper.citations or 0}",
                f"Abstract: {paper.abstract or 'No abstract available'}",
                f"Fields of Study: {', '.join(paper.fieldsOfStudy or [])}",
                f"URL: {paper.url or 'No URL available'}",
            ]

            return "\n".join(response)

        except Exception as e:
            return f"Error fetching paper details: {str(e)}"


# Example usage:
"""
tool = SemanticScholarTool()
results = await tool.search_papers("machine learning", year_start=2020)
paper_details = await tool.get_paper_details("paper_id_here")
"""
