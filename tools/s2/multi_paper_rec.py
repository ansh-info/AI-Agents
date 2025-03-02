import time
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

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
        """Validate paper IDs list."""
        if not v:
            raise ValueError("At least one paper ID must be provided")
        if len(v) > 10:  # Arbitrary limit to prevent abuse
            raise ValueError("Maximum of 10 paper IDs allowed")
        return v


@tool(args_schema=MultiPaperRecInput, return_direct=True)
def get_multi_paper_recommendations(
    paper_ids: List[str], limit: int = 5
) -> Dict[str, Any]:
    """Get paper recommendations based on multiple papers.

    Args:
        paper_ids: List of Semantic Scholar Paper IDs to get recommendations for
        limit: Maximum total number of recommendations to return

    Returns:
        Dict containing aggregated recommendations or error information
    """
    try:
        if not paper_ids:
            return {
                "status": "error",
                "error": "At least one paper ID must be provided",
                "recommendations": [],
                "message": "At least one paper ID must be provided",
            }

        all_recommendations = []
        errors = []

        # Calculate recommendations per paper
        recs_per_paper = max(1, limit // len(paper_ids))

        for paper_id in paper_ids:
            try:
                endpoint = (
                    f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
                )
                params = {
                    "limit": min(recs_per_paper, 100)
                }  # Respect unauthenticated limit

                max_retries = 3
                retry_delay = 2  # Starting delay for unauthenticated access

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
                            errors.append(f"Paper with ID {paper_id} not found.")
                            break
                        elif response.status_code == 400:
                            errors.append(f"Invalid paper ID format: {paper_id}")
                            break

                        response.raise_for_status()
                        data = response.json()
                        recommendations = data.get("recommendations", [])

                        if recommendations:
                            all_recommendations.extend(recommendations)
                        break  # Successful, break retry loop

                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        errors.append(
                            f"Failed to get recommendations for paper {paper_id}: {str(e)}"
                        )
                        break

            except Exception as e:
                errors.append(f"Error processing paper {paper_id}: {str(e)}")

        if not all_recommendations and errors:
            return {
                "status": "error",
                "error": "; ".join(errors),
                "recommendations": [],
                "message": f"Failed to get any recommendations. Errors: {'; '.join(errors)}",
            }

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
            "recommendations": final_recommendations,
            "total": len(final_recommendations),
            "errors": errors if errors else None,
            "message": (
                f"Found {len(final_recommendations)} unique recommendations "
                f"across {len(paper_ids)} papers."
                + (f" Some errors occurred: {'; '.join(errors)}" if errors else "")
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "recommendations": [],
            "message": f"Error during multi-paper recommendation process: {str(e)}",
        }
