import time
from typing import Any, Dict

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SinglePaperRecInput(BaseModel):
    paper_id: str = Field(description="ID of the paper to get recommendations for")
    limit: int = Field(default=5, description="Maximum number of recommendations")


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(paper_id: str, limit: int = 5) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper."""
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
    params = {"limit": limit}

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params)

            if response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

            response.raise_for_status()
            data = response.json()
            recommendations = data.get("recommendations", [])

            # Update shared state
            shared_state.add_papers(recommendations)

            return {
                "status": "success",
                "recommendations": recommendations,
                "total": len(recommendations),
            }

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                error_msg = f"Error getting recommendations: {str(e)}"
                shared_state.set(config.StateKeys.ERROR, error_msg)
                return {"status": "error", "error": error_msg, "recommendations": []}
            time.sleep(retry_delay)
            retry_delay *= 2
