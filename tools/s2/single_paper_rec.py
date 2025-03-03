import time
from typing import Any, Dict, Optional
import requests
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field, field_validator
import re

from config.config import config
from state.shared_state import shared_state


class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for (40-character string)"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of recommendations to return",
        ge=1,
        le=100,
    )

    @field_validator("paper_id")
    def validate_paper_id(cls, v: str) -> str:
        """Validate paper ID format (40-character string)."""
        if not re.match(r"^[a-f0-9]{40}$", v):
            raise ValueError("Paper ID must be a 40-character hexadecimal string")
        return v


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(paper_id: str, limit: int = 5) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper.

    Best for:
    - Finding papers similar to a specific paper
    - Getting research recommendations
    - Finding related work

    The paper_id must be a 40-character Semantic Scholar ID.

    Examples:
    - Finding similar papers to a specific paper
    - Getting recommendations in the same research area
    - Finding papers that build on a specific paper

    Args:
        paper_id: Semantic Scholar Paper ID
        limit: Maximum number of recommendations to return

    Returns:
        Dict containing recommended papers or error information
    """
    try:
        endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
        params = {"limit": min(limit, 100)}  # Respect unauthenticated limit

        max_retries = 3
        retry_delay = 2  # Starting delay

        for attempt in range(max_retries):
            try:
                response = requests.get(endpoint, params=params)

                if response.status_code == 429:  # Rate limit
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue

                if response.status_code == 404:
                    raise ToolException(f"Paper with ID {paper_id} not found.")

                response.raise_for_status()
                data = response.json()
                recommendations = data.get("recommendations", [])

                # Update shared state
                if recommendations:
                    shared_state.add_papers(recommendations)

                return {
                    "status": "success",
                    "papers": recommendations,
                    "total": len(recommendations),
                    "message": f"Found {len(recommendations)} recommended papers.",
                }

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                raise ToolException(f"Error getting recommendations: {str(e)}")

        raise ToolException(f"Failed after {max_retries} attempts.")

    except Exception as e:
        raise ToolException(f"Error getting recommendations: {str(e)}")
