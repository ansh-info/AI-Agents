from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.semantic_scholar_client import (PaperMetadata, SearchFilters,
                                             SemanticScholarClient)


class SearchPapersSchema(BaseModel):
    """Schema for search_papers parameters"""

    query: str = Field(..., description="The search query for finding papers")
    year_start: Optional[int] = Field(None, description="Start year filter")
    year_end: Optional[int] = Field(None, description="End year filter")
    min_citations: Optional[int] = Field(None, description="Minimum citations")


class SemanticScholarTool(BaseTool):
    """Tool for Semantic Scholar interactions"""

    name: str = "semantic_scholar_tool"
    description: str = "Tool for searching and retrieving academic papers"
    args_schema: Type[BaseModel] = SearchPapersSchema
    _client: SemanticScholarClient = (
        PrivateAttr()
    )  # Use PrivateAttr for non-serialized fields

    def __init__(self):
        """Initialize the tool"""
        super().__init__()
        self._client = SemanticScholarClient()

    async def _arun(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> str:
        """Asynchronously search for papers."""
        try:
            filters = SearchFilters(
                year_start=year_start, year_end=year_end, min_citations=min_citations
            )

            results = await self._client.search_papers(query=query, filters=filters)

            return self._format_search_results(results)
        except Exception as e:
            return f"Error searching papers: {str(e)}"

    @tool
    async def search_papers(
        self,
        query: Annotated[str, "The search query for finding papers"],
        year_start: Annotated[Optional[int], "Start year filter"] = None,
        year_end: Annotated[Optional[int], "End year filter"] = None,
        min_citations: Annotated[Optional[int], "Minimum citations"] = None,
        is_open_access: Annotated[
            Optional[bool], "Filter for open access papers"
        ] = None,
    ) -> str:
        """Search for academic papers on Semantic Scholar."""
        try:
            filters = SearchFilters(
                year_start=year_start,
                year_end=year_end,
                min_citations=min_citations,
                is_open_access=is_open_access,
            )

            results = await self.client.search_papers(query=query, filters=filters)

            return self._format_search_results(results)

        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
            return f"Error searching papers: {str(e)}"

            # Format results for agent consumption
            papers_info = []
            for i, paper in enumerate(results.papers, 1):
                paper_info = {
                    "number": i,
                    "id": paper.paperId,
                    "title": paper.title,
                    "authors": [
                        {"name": a.name, "id": a.authorId} for a in paper.authors
                    ],
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

            # Build response
            response_parts = [
                f"Found {results.total} papers related to '{query}'. Here are the most relevant papers:\n"
            ]

            for paper in papers_info:
                paper_text = [
                    f"\n[{paper['number']}] {paper['title']}",
                    f"Authors: {', '.join(a['name'] for a in paper['authors'])}",
                    f"Year: {paper['year'] or 'N/A'} | Citations: {paper['citations'] or 0}",
                    f"Abstract: {paper['abstract']}",
                    f"URL: {paper['url'] or 'No URL available'}\n",
                ]
                response_parts.extend(paper_text)

            return "\n".join(response_parts)

        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
            return f"Error searching papers: {str(e)}"

    @tool
    async def get_paper_details(
        self,
        paper_id: Annotated[str, "ID of the paper to fetch details for"],
    ) -> str:
        """
        Get detailed information about a specific paper including full metadata,
        citation count, and field classifications.
        """
        try:
            print(f"[DEBUG] Fetching paper details for ID: {paper_id}")

            paper = await self.client.get_paper_details(paper_id)

            # Format detailed response
            response = [
                f"Title: {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'}",
                f"Citations: {paper.citations or 0}",
                f"Fields of Study: {', '.join(paper.fieldsOfStudy) if paper.fieldsOfStudy else 'Not specified'}",
                "",
                "Abstract:",
                f"{paper.abstract or 'No abstract available'}",
                "",
                f"URL: {paper.url or 'No URL available'}",
                "",
                "Additional Information:",
                f"- Open Access: {'Yes' if paper.isOpenAccess else 'No'}",
            ]

            return "\n".join(response)

        except Exception as e:
            print(f"[DEBUG] Error fetching paper details: {str(e)}")
            return f"Error fetching paper details: {str(e)}"

    @tool
    async def search_by_filters(
        self,
        filters: Annotated[Dict[str, Any], "Dictionary of search filters"],
        limit: Annotated[int, "Maximum number of results"] = 10,
    ) -> str:
        """
        Advanced search using custom filters. Accepts a dictionary of filters and returns
        matching papers. Useful for complex queries with multiple criteria.
        """
        try:
            print(f"[DEBUG] Searching with filters: {filters}")

            # Convert dictionary to SearchFilters object
            search_filters = SearchFilters(**filters)

            # Perform search
            results = await self.client.search_papers(
                query=filters.get("query", ""), filters=search_filters, limit=limit
            )

            # Format and return results
            return self._format_search_results(results)

        except Exception as e:
            print(f"[DEBUG] Filter search error: {str(e)}")
            return f"Error in filtered search: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        """Synchronously search (not implemented)."""
        raise NotImplementedError("This tool only supports async execution")

    def _format_search_results(self, results) -> str:
        """Format search results into a readable string."""
        if not results.papers:
            return "No papers found matching the criteria."

        response_parts = [
            f"Found {results.total} papers. Showing top {len(results.papers)} results:\n"
        ]

        for i, paper in enumerate(results.papers, 1):
            paper_text = [
                f"\n[{i}] {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}",
                f"Abstract: {paper.abstract[:300] + '...' if paper.abstract else 'No abstract available'}",
                f"URL: {paper.url or 'No URL available'}\n",
            ]
            response_parts.extend(paper_text)

        return "\n".join(response_parts)
