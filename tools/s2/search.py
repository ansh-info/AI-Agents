import time
from typing import Any, Dict, List, Optional, Type
import requests
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=5, description="Maximum number of results to return")
    fields: Optional[List[str]] = Field(
        default=None, description="List of fields to include in results"
    )


@tool(args_schema=SearchInput)
def search_papers(
    query: str, limit: int = 5, fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search for academic papers on Semantic Scholar."""
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

    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"
    params = {"query": query, "limit": limit, "fields": ",".join(fields)}

    max_retries = 3
    retry_delay = 1  # Starting delay in seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={
                    "x-api-key": config.SEMANTIC_SCHOLAR_API_KEY
                },  # Add API key if available
            )

            if response.status_code == 429:  # Rate limit hit
                if attempt < max_retries - 1:  # If not the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])

            # Filter out papers with missing crucial information
            filtered_papers = [
                paper
                for paper in papers
                if paper.get("title") and paper.get("authors")  # Basic validation
            ]

            # Update shared state with search results
            shared_state.add_papers(filtered_papers)

            return {
                "status": "success",
                "papers": filtered_papers,
                "total": len(filtered_papers),
            }

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # If last attempt
                error_msg = f"Error searching papers: {str(e)}"
                shared_state.set(config.StateKeys.ERROR, error_msg)
                return {"status": "error", "error": error_msg, "papers": []}
            time.sleep(retry_delay)
            retry_delay *= 2
