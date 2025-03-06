import time
from typing import Annotated, Any, Dict, List

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
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
        default=2,
        description="Maximum total number of recommendations to return",
        ge=1,
        le=100,
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

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
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
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
    print("Starting multi-paper recommendations search...")

    if not paper_ids:
        raise ToolException("At least one paper ID must be provided")

    all_recommendations = []
    errors = []

    # Calculate recommendations per paper
    recs_per_paper = max(1, limit // len(paper_ids))

    for paper_id in paper_ids:
        print(f"Processing paper ID: {paper_id}")
        endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
        params = {"limit": min(recs_per_paper, 100)}

        max_retries = 3
        retry_count = 0
        retry_delay = 2

        while retry_count < max_retries:
            try:
                print(f"Attempt {retry_count + 1} of {max_retries}")
                response = requests.get(endpoint, params=params, timeout=10)

                if response.status_code == 429:  # Rate limit hit
                    retry_count += 1
                    wait_time = retry_delay * (2**retry_count)
                    print(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 404:
                    errors.append(f"Paper with ID {paper_id} not found.")
                    break

                # Break immediately if we get a successful response
                if response.status_code == 200:
                    print(f"Successful response received for paper {paper_id}")
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    if recommendations:
                        all_recommendations.extend(recommendations)
                    break

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                print(f"Request failed for paper {paper_id}: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    errors.append(f"Error processing paper {paper_id}: {str(e)}")
                    break
                time.sleep(retry_delay * (2**retry_count))
                continue

    if not all_recommendations and errors:
        raise ToolException(
            f"Failed to get any recommendations. Errors: {'; '.join(errors)}"
        )

    print("Processing aggregated recommendations...")

    # Deduplicate recommendations
    seen = set()
    unique_recommendations = []
    for rec in all_recommendations:
        if rec.get("paperId") not in seen:
            seen.add(rec.get("paperId"))
            unique_recommendations.append(rec)

    # Take top limit recommendations
    final_recommendations = unique_recommendations[:limit]

    # Create DataFrame
    filtered_papers = [
        (paper["paperId"], paper["title"])
        for paper in final_recommendations
        if paper.get("title")
    ]

    df = pd.DataFrame(filtered_papers, columns=["Paper ID", "Title"])
    print("Created DataFrame with results")
    print(df)

    print("Multi-paper recommendations tool execution completed")
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
