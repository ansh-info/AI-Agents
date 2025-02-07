from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.tools import tool, BaseTool
import requests

from config.config import config
from state.shared_state import shared_state


class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=10, description="Maximum number of results to return")
    fields: Optional[List[str]] = Field(
        default=None, description="List of fields to include in results"
    )


@tool(args_schema=SearchInput)
def search_papers(
    query: str, limit: int = 10, fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search for academic papers on Semantic Scholar.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)
        fields: Optional list of fields to include in results

    Returns:
        Dict containing search results and status
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

    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"

    params = {"query": query, "limit": limit, "fields": ",".join(fields)}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        papers = data.get("data", [])

        # Update shared state with search results
        shared_state.add_papers(papers)

        return {"status": "success", "papers": papers, "total": len(papers)}

    except requests.exceptions.RequestException as e:
        error_msg = f"Error searching papers: {str(e)}"
        shared_state.set(config.StateKeys.ERROR, error_msg)
        return {"status": "error", "error": error_msg, "papers": []}


# Create schema for single paper recommendation
class SinglePaperRecInput(BaseModel):
    paper_id: str = Field(description="ID of the paper to get recommendations for")
    limit: int = Field(default=10, description="Maximum number of recommendations")


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(paper_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper.

    Args:
        paper_id: ID of the paper to get recommendations for
        limit: Maximum number of recommendations to return

    Returns:
        Dict containing recommended papers
    """
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"

    params = {"limit": limit}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        recommendations = data.get("recommendations", [])

        # Update shared state with recommendations
        shared_state.add_papers(recommendations)

        return {
            "status": "success",
            "recommendations": recommendations,
            "total": len(recommendations),
        }

    except requests.exceptions.RequestException as e:
        error_msg = f"Error getting recommendations: {str(e)}"
        shared_state.set(config.StateKeys.ERROR, error_msg)
        return {"status": "error", "error": error_msg, "recommendations": []}


# Create schema for multi-paper recommendations
class MultiPaperRecInput(BaseModel):
    paper_ids: List[str] = Field(
        description="List of paper IDs to get recommendations for"
    )
    limit: int = Field(default=10, description="Maximum number of recommendations")


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str], limit: int = 10
) -> Dict[str, Any]:
    """Get paper recommendations based on multiple papers.

    Args:
        paper_ids: List of paper IDs to get recommendations for
        limit: Maximum number of recommendations to return

    Returns:
        Dict containing recommended papers
    """
    # Here we'd implement logic to get recommendations based on multiple papers
    # This could involve calling the API multiple times and aggregating results
    # For now, we'll use a simple implementation
    all_recommendations = []

    try:
        for paper_id in paper_ids:
            result = get_single_paper_recommendations(
                paper_id, limit=limit // len(paper_ids)
            )
            if result["status"] == "success":
                all_recommendations.extend(result["recommendations"])

        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec["paperId"] not in seen:
                seen.add(rec["paperId"])
                unique_recommendations.append(rec)

        # Take top limit recommendations
        final_recommendations = unique_recommendations[:limit]

        # Update shared state with recommendations
        shared_state.add_papers(final_recommendations)

        return {
            "status": "success",
            "recommendations": final_recommendations,
            "total": len(final_recommendations),
        }

    except Exception as e:
        error_msg = f"Error getting multi-paper recommendations: {str(e)}"
        shared_state.set(config.StateKeys.ERROR, error_msg)
        return {"status": "error", "error": error_msg, "recommendations": []}


# Export all tools
s2_tools = [
    search_papers,
    get_single_paper_recommendations,
    get_multi_paper_recommendations,
]
