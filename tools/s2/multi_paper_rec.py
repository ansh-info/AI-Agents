from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from config.config import config
from state.shared_state import shared_state
from tools.s2.single_paper_rec import get_single_paper_recommendations


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
            result = get_single_paper_recommendations(
                paper_id=paper_id, limit=recs_per_paper
            )
            if result["status"] == "success":
                all_recommendations.extend(result["recommendations"])
            else:
                errors.append(f"Error for paper {paper_id}: {result.get('error')}")

        if not all_recommendations and errors:
            error_msg = "; ".join(errors)
            return {
                "status": "error",
                "error": f"Failed to get any recommendations: {error_msg}",
                "recommendations": [],
                "message": f"Failed to get any recommendations: {error_msg}",
            }

        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec["paperId"] not in seen:
                seen.add(rec["paperId"])
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
