import re
import time
from typing import Annotated, Any, Dict, List
import json
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

    @validator("paper_ids")
    def validate_paper_ids(cls, v: List[str]) -> List[str]:
        """Validate paper IDs format"""
        if not v:
            raise ValueError("At least one paper ID must be provided")
        if len(v) > 10:
            raise ValueError("Maximum of 10 paper IDs allowed")
        return v

    model_config = {"arbitrary_types_allowed": True}


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """Get paper recommendations based on multiple papers."""
    logging.info("Starting multi-paper recommendations search.")

    endpoint = "https://api.semanticscholar.org/recommendations/v1/papers"
    headers = {"Content-Type": "application/json"}
    payload = {"positivePaperIds": paper_ids, "negativePaperIds": []}
    params = {
        "limit": min(limit, 500),
        "fields": "title,paperId,abstract,year,url,publicationTypes,openAccessPdf",
    }

    response = requests.post(
        endpoint, headers=headers, params=params, data=json.dumps(payload), timeout=10
    )
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
