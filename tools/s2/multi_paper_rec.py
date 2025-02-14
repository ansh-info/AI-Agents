from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state
from tools.s2.single_paper_rec import get_single_paper_recommendations


class MultiPaperRecInput(BaseModel):
    paper_ids: List[str] = Field(
        description="List of paper IDs to get recommendations for"
    )
    limit: int = Field(default=5, description="Maximum number of recommendations")


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str], limit: int = 5
) -> Dict[str, Any]:
    """Get paper recommendations based on multiple papers."""
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

        # Update shared state
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
