import time
from typing import Annotated, Any, Dict, Optional

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
import re

from state.shared_state import Talk2Papers
from config.config import config
from state.shared_state import shared_state


class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for (40-character string)"
    )
    limit: int = Field(
        default=2,
        description="Maximum number of recommendations to return",
        ge=1,
        le=100,
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    @field_validator("paper_id")
    def validate_paper_id(cls, v: str) -> str:
        """Validate paper ID format (40-character string)."""
        if not re.match(r"^[a-f0-9]{40}$", v):
            raise ValueError("Paper ID must be a 40-character hexadecimal string")
        return v


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
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
    print("Starting single paper recommendations search...")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
    params = {"limit": min(limit, 100)}

    max_retries = 3
    retry_count = 0
    retry_delay = 2

    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1} of {max_retries}")
            response = requests.get(endpoint, params=params, timeout=10)

            if response.status_code == 429:  # Rate limit hit
                retry_count += 1
                wait_time = retry_delay * (2**retry_count)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if response.status_code == 404:
                raise ToolException(f"Paper with ID {paper_id} not found.")

            # Break immediately if we get a successful response
            if response.status_code == 200:
                print("Successful response received")
                break

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                raise ToolException(
                    f"Error getting recommendations after {max_retries} attempts: {str(e)}"
                )
            time.sleep(retry_delay * (2**retry_count))
            continue

    print("Processing response...")
    data = response.json()
    recommendations = data.get("recommendations", [])

    # Create DataFrame
    filtered_papers = [
        (paper["paperId"], paper["title"])
        for paper in recommendations
        if paper.get("title")
    ]

    df = pd.DataFrame(filtered_papers, columns=["Paper ID", "Title"])
    print("Created DataFrame with results")
    print(df)

    print("Single paper recommendations tool execution completed")
    return Command(
        update={
            "papers": df.to_markdown(tablefmt="grid"),
            "messages": [
                ToolMessage(
                    content=df.to_markdown(tablefmt="grid"),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
