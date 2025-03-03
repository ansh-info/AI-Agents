import time
from typing import Any, Dict, List, Optional
import requests
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers. Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=5, description="Maximum number of results to return", ge=1, le=100
    )


@tool(args_schema=SearchInput)
def search_papers(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search for academic papers on Semantic Scholar.

    Best for:
    - Finding papers on specific topics
    - Academic research queries
    - Finding recent papers in a field

    Examples:
    - "machine learning applications in healthcare"
    - "recent advances in transformers 2023"
    - "quantum computing algorithms review"

    Args:
        query: Search query string
        limit: Maximum number of results to return (max 100)

    Returns:
        Dict containing search results or error information
    """
    try:
        endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),  # Respect unauthenticated limit
            "fields": "paperId,title,abstract,year,authors,citationCount,openAccessPdf",
        }

        max_retries = 3
        retry_delay = 2  # Starting delay in seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(endpoint, params=params)

                if response.status_code == 429:  # Rate limit hit
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()
                papers = data.get("data", [])

                # Filter and clean results
                filtered_papers = [
                    paper
                    for paper in papers
                    if paper.get("title") and paper.get("authors")
                ]

                # Update shared state
                shared_state.add_papers(filtered_papers)

                return {
                    "status": "success",
                    "papers": filtered_papers,
                    "total": len(filtered_papers),
                    "message": f"Found {len(filtered_papers)} papers matching your query.",
                }

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                raise ToolException(f"Error searching papers: {str(e)}")

        raise ToolException(f"Failed after {max_retries} attempts.")

    except Exception as e:
        raise ToolException(f"Error during search process: {str(e)}")
