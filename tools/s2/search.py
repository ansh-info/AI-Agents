from typing import Any, Dict, List

import requests

from config.config import config
from state.shared_state import shared_state


class SemanticScholarSearchTool:
    def __init__(self):
        self.base_url = config.SEMANTIC_SCHOLAR_API

    def search_papers(
        self, query: str, limit: int = 10, fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search for papers on Semantic Scholar

        Args:
            query: Search query string
            limit: Maximum number of results (default: 10)
            fields: List of fields to include in results

        Returns:
            Dict containing search results and status
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "citationCount",
                "openAccessPdf",
            ]

        endpoint = f"{self.base_url}/paper/search"

        params = {"query": query, "limit": limit, "fields": ",".join(fields)}

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            papers = data.get("data", [])

            # Update shared state with search results
            shared_state.set(config.StateKeys.PAPERS, papers)

            return {"status": "success", "papers": papers, "total": len(papers)}

        except requests.exceptions.RequestException as e:
            error_msg = f"Error searching papers: {str(e)}"
            shared_state.set(config.StateKeys.ERROR, error_msg)
            return {"status": "error", "error": error_msg, "papers": []}


# Create a global instance
s2_search_tool = SemanticScholarSearchTool()
