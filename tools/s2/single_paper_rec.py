import time
from typing import Any, Dict

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of recommendations to return",
        ge=1,
        le=100,
    )


@tool(args_schema=SinglePaperRecInput, return_direct=True)
def get_single_paper_recommendations(paper_id: str, limit: int = 5) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper.

    Args:
        paper_id: Semantic Scholar Paper ID to get recommendations for
        limit: Maximum number of recommendations to return (1-100)

    Returns:
        Dict containing recommended papers or error information
    """
    try:
        endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
        params = {"limit": min(limit, 100)}  # Respect unauthenticated limit

        max_retries = 3
        retry_delay = 2  # Starting delay for unauthenticated access
        last_error = None

        for attempt in range(max_retries):
            try:
                # Make request without API key
                response = requests.get(endpoint, params=params)

                if response.status_code == 429:  # Rate limit hit
                    wait_time = 2**attempt  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                # Handle various error cases
                if response.status_code == 404:
                    return {
                        "status": "error",
                        "error": f"Paper with ID {paper_id} not found.",
                        "recommendations": [],
                        "message": f"Paper with ID {paper_id} not found.",
                    }
                elif response.status_code == 400:
                    return {
                        "status": "error",
                        "error": f"Invalid paper ID format: {paper_id}",
                        "recommendations": [],
                        "message": f"Invalid paper ID format: {paper_id}",
                    }

                response.raise_for_status()
                data = response.json()
                recommendations = data.get("recommendations", [])

                # Validate recommendations
                if not recommendations:
                    return {
                        "status": "success",
                        "recommendations": [],
                        "total": 0,
                        "message": f"No recommendations found for paper {paper_id}.",
                    }

                # Update shared state
                shared_state.add_papers(recommendations)

                return {
                    "status": "success",
                    "recommendations": recommendations,
                    "total": len(recommendations),
                    "message": f"Found {len(recommendations)} recommended papers.",
                }

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if response.status_code == 429:  # Rate limit
                    wait_time = 2**attempt
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                return {
                    "status": "error",
                    "error": f"Error getting recommendations: {last_error}",
                    "recommendations": [],
                    "message": f"Error getting recommendations: {last_error}",
                }

        return {
            "status": "error",
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "recommendations": [],
            "message": f"Failed after {max_retries} attempts. Last error: {last_error}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "recommendations": [],
            "message": f"Error getting recommendations: {str(e)}",
        }
