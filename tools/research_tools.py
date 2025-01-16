from typing import Annotated, Any, Dict, List, Optional

from agent_state import AgentState
from langchain_core.tools import tool
from semantic_scholar_client import SearchFilters, SemanticScholarClient


class ResearchTools:
    """Collection of research-related tools for the agent."""

    def __init__(self, s2_client: SemanticScholarClient):
        self.s2_client = s2_client

    @tool
    async def search_papers(
        self,
        query: Annotated[str, "Search query for academic papers"],
        year_start: Annotated[Optional[int], "Start year filter"] = None,
        year_end: Annotated[Optional[int], "End year filter"] = None,
        min_citations: Annotated[Optional[int], "Minimum citations filter"] = None,
    ) -> str:
        """Search for academic papers using Semantic Scholar."""
        try:
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            results = await self.s2_client.search_papers(query=query, filters=filters)

            # Format results for the agent
            papers_info = []
            for i, paper in enumerate(results.papers, 1):
                paper_info = {
                    "number": i,
                    "title": paper.title,
                    "authors": ", ".join(a.name for a in paper.authors),
                    "year": paper.year,
                    "citations": paper.citations,
                    "abstract": (
                        paper.abstract[:300] + "..."
                        if paper.abstract
                        else "No abstract available"
                    ),
                    "url": paper.url,
                }
                papers_info.append(paper_info)

            return (
                f"Found {results.total} papers. Here are the most relevant ones:\n\n"
                + "\n\n".join(
                    [
                        f"[{p['number']}] {p['title']}\n"
                        + f"Authors: {p['authors']}\n"
                        + f"Year: {p['year']}, Citations: {p['citations']}\n"
                        + f"Abstract: {p['abstract']}\n"
                        + f"URL: {p['url']}"
                        for p in papers_info
                    ]
                )
            )
        except Exception as e:
            return f"Error searching papers: {str(e)}"

    @tool
    async def get_paper_details(
        self, paper_id: Annotated[str, "The ID of the paper to retrieve details for"]
    ) -> str:
        """Get detailed information about a specific paper."""
        try:
            paper = await self.s2_client.get_paper_details(paper_id)
            return f"""
Title: {paper.title}
Authors: {', '.join(a.name for a in paper.authors)}
Year: {paper.year or 'N/A'}
Citations: {paper.citations or 0}
Abstract: {paper.abstract or 'No abstract available'}
URL: {paper.url or 'No URL available'}
"""
        except Exception as e:
            return f"Error getting paper details: {str(e)}"
