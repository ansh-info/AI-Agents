import time
from typing import Any, Dict, List
import requests
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field, field_validator
import re

from config.config import config
from state.shared_state import shared_state


class MultiPaperRecInput(BaseModel):
    """Input schema for multiple paper recommendations tool."""

    paper_ids: List[str] = Field(
        description="List of Semantic Scholar Paper IDs to get recommendations for"
    )
    limit: int = Field(
        default=5,
        description="Maximum total number of recommendations to return",
        ge=1,
        le=100,
    )

    @field_validator("paper_ids")
    def validate_paper_ids(cls, v: List[str]) -> List[str]:
        """Validate paper IDs format."""
        if not v:
            raise ValueError("At least one paper ID must be provided")
        if len(v) > 10:  # Arbitrary limit to prevent abuse
            raise ValueError("Maximum of 10 paper IDs allowed")
        for paper_id in v:
            if not re.match(r"^[a-f0-9]{40}$", paper_id):
                raise ValueError(f"Invalid paper ID format: {paper_id}")
        return v


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str], limit: int = 5
) -> Dict[str, Any]:
    """Get paper recommendations based on multiple papers.

    Best for:
    - Finding papers related to multiple seed papers
    - Getting recommendations across multiple research areas
    - Finding papers at the intersection of multiple topics

    Each paper_id must be a 40-character Semantic Scholar ID.

    Examples:
    - Finding papers similar to a set of papers
    - Getting recommendations that bridge multiple research areas
    - Finding papers related to multiple aspects of a topic

    Args:
        paper_ids: List of Semantic Scholar Paper IDs
        limit: Maximum total number of recommendations to return

    Returns:
        Dict containing aggregated recommendations or error information
    """
    try:
        if not paper_ids:
            raise ToolException("At least one paper ID must be provided")

        all_recommendations = []
        errors = []

        # Calculate recommendations per paper
        recs_per_paper = max(1, limit // len(paper_ids))

        for paper_id in paper_ids:
            try:
                endpoint = (
                    f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
                )
                params = {"limit": min(recs_per_paper, 100)}

                max_retries = 3
                retry_delay = 2

                for attempt in range(max_retries):
                    try:
                        response = requests.get(endpoint, params=params)

                        if response.status_code == 429:  # Rate limit
                            wait_time = 2**attempt
                            time.sleep(wait_time)
                            continue

                        if response.status_code == 404:
                            errors.append(f"Paper with ID {paper_id} not found.")
                            break

                        response.raise_for_status()
                        data = response.json()
                        recommendations = data.get("recommendations", [])

                        if recommendations:
                            all_recommendations.extend(recommendations)
                        break  # Success

                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        errors.append(f"Error processing paper {paper_id}: {str(e)}")
                        break

            except Exception as e:
                errors.append(f"Error processing paper {paper_id}: {str(e)}")

        if not all_recommendations and errors:
            raise ToolException(
                f"Failed to get any recommendations. Errors: {'; '.join(errors)}"
            )

        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec.get("paperId") not in seen:
                seen.add(rec.get("paperId"))
                unique_recommendations.append(rec)

        # Take top limit recommendations
        final_recommendations = unique_recommendations[:limit]

        # Update shared state
        shared_state.add_papers(final_recommendations)

        return {
            "status": "success",
            "papers": final_recommendations,
            "total": len(final_recommendations),
            "errors": errors if errors else None,
            "message": (
                f"Found {len(final_recommendations)} unique recommendations "
                f"across {len(paper_ids)} papers."
                + (f" Some errors occurred: {'; '.join(errors)}" if errors else "")
            ),
        }

    except Exception as e:
        raise ToolException(
            f"Error during multi-paper recommendation process: {str(e)}"
        )
