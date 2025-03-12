import re
import time
from typing import Annotated, Any, Dict

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator


class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for (40-character string)"
    )
    limit: int = Field(
        default=2,
        description="Maximum number of recommendations to return",
        ge=1,
        le=500,
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

    endpoint = (
        f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
    )
    params = {
        "limit": min(limit, 500),
        "fields": "title,paperId,abstract,year",
        "from": "all-cs",
    }

    max_retries = 3
    retry_count = 0
    retry_delay = 2

    while retry_count < max_retries:
        print(f"Attempt {retry_count + 1} of {max_retries}")
        response = requests.get(endpoint, params=params)
        print(f"API Response Status: {response.status_code}")
        print(f"Request params: {params}")

        if response.status_code == 200:
            data = response.json()
            print(f"Raw API Response: {data}")
            recommendations = data.get("recommendedPapers", [])

            if recommendations:
                filtered_papers = [
                    {"Paper ID": paper["paperId"], "Title": paper["title"]}
                    for paper in recommendations
                    if paper.get("title") and paper.get("paperId")
                ]

                if filtered_papers:
                    df = pd.DataFrame(filtered_papers)

                    papers = [
                        f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
                        for paper in filtered_papers
                    ]

                    markdown_table = df.to_markdown(tablefmt="grid")

                    return Command(
                        update={
                            "papers": papers,
                            "messages": [
                                ToolMessage(
                                    content=markdown_table, tool_call_id=tool_call_id
                                )
                            ],
                        }
                    )

            return Command(
                update={
                    "papers": [],
                    "messages": [
                        ToolMessage(
                            content="No recommendations found for this paper",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        retry_count += 1
        if retry_count < max_retries:
            wait_time = retry_delay * (2**retry_count)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise ToolException(
        f"Error getting recommendations after {max_retries} attempts. Status code: {response.status_code}"
    )
