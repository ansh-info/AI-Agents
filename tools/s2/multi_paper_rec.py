import re
from typing import Annotated, Any, Dict, List
import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
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

    # Correct recommendations API endpoint for multiple papers
    endpoint = "https://api.semanticscholar.org/recommendations/v1/papers"
    headers = {"Content-Type": "application/json"}
    data = {"positivePaperIds": paper_ids, "negativePaperIds": []}
    params = {"limit": min(limit, 500), "fields": "title,paperId"}

    print(f"Calling endpoint: {endpoint}")
    response = requests.post(endpoint, json=data, headers=headers, params=params)
    print(f"API Response Status: {response.status_code}")
    print(f"Request data: {data}")

    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response: {data}")
        recommendations = data.get("recommendedPapers", [])

        if not recommendations:
            print("No recommendations found")
            return Command(
                update={
                    "papers": "No recommendations found for the provided papers",
                    "messages": [
                        ToolMessage(
                            content="No recommendations found for the provided papers",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # Create DataFrame
        filtered_papers = [
            (paper["paperId"], paper["title"])
            for paper in recommendations
            if paper.get("title") and paper.get("paperId")
        ]

        if not filtered_papers:
            return Command(
                update={
                    "papers": "No valid recommendations found",
                    "messages": [
                        ToolMessage(
                            content="No valid recommendations found",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        df = pd.DataFrame(filtered_papers, columns=["Paper ID", "Title"])
        print("Created DataFrame with results:")
        print(df)

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
    elif response.status_code == 404:
        error_msg = "One or more papers not found"
        raise ToolException(error_msg)
    else:
        error_msg = (
            f"Error getting recommendations. Status code: {response.status_code}"
        )
        raise ToolException(error_msg)
