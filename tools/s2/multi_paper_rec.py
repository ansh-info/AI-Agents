import re
import time
from typing import Annotated, Any, Dict, List

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator


class MultiPaperRecInput(BaseModel):
    """Input schema for multiple paper recommendations tool."""

    paper_ids: List[str] = Field(
        description="List of Semantic Scholar Paper IDs to get recommendations for"
    )
    limit: int = Field(
        default=2,
        description="Maximum total number of recommendations to return",
        ge=1,
        le=500,
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    @field_validator("paper_ids")
    def validate_paper_ids(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one paper ID must be provided")
        if len(v) > 10:
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
    """Get paper recommendations based on multiple papers."""
    print("Starting multi-paper recommendations search...")

    all_recommendations = []
    paper_results = []

    # Get recommendations for each paper ID
    for paper_id in paper_ids:
        endpoint = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        params = {"limit": min(limit, 500), "fields": "title,paperId"}

        max_retries = 3
        retry_count = 0
        retry_delay = 2

        while retry_count < max_retries:
            print(f"Attempt {retry_count + 1} of {max_retries} for paper {paper_id}")
            response = requests.get(endpoint, params=params)
            print(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendedPapers", [])
                all_recommendations.extend(recommendations)
                break

            retry_count += 1
            if retry_count < max_retries:
                wait_time = retry_delay * (2**retry_count)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to get recommendations for paper {paper_id}")

    if not all_recommendations:
        return Command(
            update={
                "papers": ["No recommendations found for the provided papers"],
                "messages": [
                    ToolMessage(
                        content="No recommendations found for the provided papers",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Create DataFrame from all recommendations
    filtered_papers = [
        (paper["paperId"], paper["title"])
        for paper in all_recommendations
        if paper.get("title") and paper.get("paperId")
    ]

    if not filtered_papers:
        return Command(
            update={
                "papers": ["No valid recommendations found"],
                "messages": [
                    ToolMessage(
                        content="No valid recommendations found",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    df = pd.DataFrame(filtered_papers, columns=["Paper ID", "Title"])
    print("Created DataFrame with all results:")
    print(df)

    # Convert each row to a formatted string
    paper_results = [
        f"Paper ID: {row['Paper ID']}\nTitle: {row['Title']}"
        for _, row in df.iterrows()
    ]

    markdown_table = df.to_markdown(tablefmt="grid")

    return Command(
        update={
            "papers": paper_results,  # Now returns a list of strings
            "messages": [
                ToolMessage(content=markdown_table, tool_call_id=tool_call_id)
            ],
        }
    )
