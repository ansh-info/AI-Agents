import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(description="Search query string to find academic papers")
    limit: int = Field(default=5, description="Maximum number of results to return")
    fields: Optional[List[str]] = Field(
        default=None, description="List of fields to include in results"
    )


def _handle_search_error(error: ToolException) -> Dict[str, Any]:
    """Handle tool execution errors in a structured way."""
    return {
        "status": "error",
        "error": str(error),
        "papers": [],
        "message": f"Failed to search papers: {str(error)}",
    }


@tool(
    args_schema=SearchInput, handle_tool_error=_handle_search_error, return_direct=True
)
def search_papers(
    query: str, limit: int = 5, fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search for academic papers on Semantic Scholar.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        fields: List of fields to include in results

    Returns:
        Dict containing search results or error information
    """
    if fields is None:
        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "citationCount",
            "openAccessPdf",
        ]

    # Clean and enhance the query
    search_terms = query.lower().split()
    if "in" in search_terms:
        search_terms.remove("in")
    if "published" in search_terms:
        search_terms.remove("published")

    # Extract year if present
    year = None
    for i, term in enumerate(search_terms):
        if term.isdigit() and len(term) == 4:
            year = int(term)
            search_terms.pop(i)
            break

    # Build enhanced query
    enhanced_query = " ".join(search_terms)
    if year:
        enhanced_query = f"year:{year} {enhanced_query}"

    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": enhanced_query,
        "limit": limit,
        "fields": ",".join(fields),
    }

    max_retries = 3
    retry_delay = 1  # Starting delay in seconds
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY},
            )

            if response.status_code == 429:  # Rate limit hit
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise ToolException("Rate limit exceeded. Please try again later.")

            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])

            # Filter invalid or incomplete papers
            filtered_papers = [
                paper
                for paper in papers
                if paper.get("title")  # Must have title
                and paper.get("authors")  # Must have authors
                and (not year or paper.get("year") == year)  # Match year if specified
            ]

            # Update shared state with search results
            shared_state.add_papers(filtered_papers)

            return {
                "status": "success",
                "papers": filtered_papers,
                "total": len(filtered_papers),
                "message": f"Found {len(filtered_papers)} papers matching your query.",
            }

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            error_msg = f"Error searching papers: {last_error}"
            raise ToolException(error_msg)

    # This should never be reached due to the exception above
    raise ToolException(
        f"Failed after {max_retries} attempts. Last error: {last_error}"
    )
