import re
from typing import Annotated, Any, Dict
import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from config.config import config


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
        if not re.match(r"^[a-f0-9]{40}$", v):
            raise ValueError("Paper ID must be a 40-character hexadecimal string")
        return v


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper."""
    print("Starting single paper recommendations search...")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
    params = {"limit": min(limit, 100)}

    response = requests.get(endpoint, params=params, timeout=10)
    print(f"API Response Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response: {data}")
        recommendations = data.get("recommendations", [])

        if not recommendations:
            print("No recommendations found")
            return Command(
                update={
                    "papers": "No recommendations found for this paper",
                    "messages": [
                        ToolMessage(
                            content="No recommendations found for this paper",
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
    else:
        error_msg = (
            f"Error getting recommendations. Status code: {response.status_code}"
        )
        raise ToolException(error_msg)
