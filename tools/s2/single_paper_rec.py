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
        description="Semantic Scholar Paper ID to get recommendations for"
    )
    limit: int = Field(
        default=2,
        description="Maximum number of recommendations to return",
        ge=1,
        le=500,
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    @validator("paper_id")
    def validate_paper_id(cls, v: str) -> str:
        """Validate paper ID format"""
        if not re.match(r"^[a-f0-9]{40}$", v):
            raise ValueError("Paper ID must be a 40-character hexadecimal string")
        return v

    model_config = {"arbitrary_types_allowed": True}


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """Get paper recommendations based on a single paper."""
    logger.info("Starting single paper recommendations search.")

    endpoint = (
        f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
    )
    params = {
        "limit": min(limit, 500),
        "fields": "title,paperId,abstract,year,url,publicationTypes,openAccessPdf",
    }

    response = requests.get(endpoint, params=params, timeout=10)
    data = response.json()
    recommendations = data.get("recommendedPapers", [])

    def get_nested_value(obj, key1, key2, default="N/A"):
        first_level = obj.get(key1)
        if first_level is not None:
            return first_level.get(key2, default)
        return default

    filtered_papers = {
        paper["paperId"]: {
            "Title": paper.get("title", "N/A"),
            "Abstract": paper.get("abstract", "N/A"),
            "Year": paper.get("year", "N/A"),
            "URL": paper.get("url", "N/A"),
            "Publication Type": (
                paper.get("publicationTypes", ["N/A"])[0]
                if paper.get("publicationTypes")
                else "N/A"
            ),
            "Open Access PDF": get_nested_value(paper, "openAccessPdf", "url"),
        }
        for paper in recommendations
        if paper.get("title")
    }

    markdown_table = pd.DataFrame(filtered_papers.values()).to_markdown(tablefmt="grid")

    return Command(
        update={
            "papers": filtered_papers,
            "messages": [
                ToolMessage(content=markdown_table, tool_call_id=tool_call_id)
            ],
        }
    )
