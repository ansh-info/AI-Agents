import time
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

from config.config import config
from state.shared_state import shared_state


class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers. Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(args_schema=SearchInput)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """Search for academic papers on Semantic Scholar.

    Best for:
    - Finding papers on specific topics
    - Academic research queries
    - Finding recent papers in a field

    Examples:
    - "machine learning applications in healthcare"
    - "recent advances in transformers 2023"
    - "quantum computing algorithms review"

    Args:
        query: Search query string
        limit: Maximum number of results to return (max 100)

    Returns:
        Dict containing search results or error information
    """
    print("Starting paper search...")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),  # Respect unauthenticated limit
        "fields": "paperId,title,abstract,year,authors,citationCount,openAccessPdf",
    }

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
                    f"Error searching papers after {max_retries} attempts: {str(e)}"
                )
            time.sleep(retry_delay * (2**retry_count))
            continue

    print("Processing response...")
    data = response.json()
    papers = data.get("data", [])

    # Filter and clean results
    filtered_papers = [
        (paper["paperId"], paper["title"])
        for paper in papers
        if paper.get("title") and paper.get("authors")
    ]

    df = pd.DataFrame(filtered_papers, columns=["Paper ID", "Title"])
    print("Created DataFrame with results")
    print(df)

    print("Search tool execution completed")
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
