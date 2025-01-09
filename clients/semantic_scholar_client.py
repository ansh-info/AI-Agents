import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Model for author information"""

    authorId: Optional[str] = None
    name: str


class PaperMetadata(BaseModel):
    """Model for paper metadata returned by Semantic Scholar API"""

    paperId: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: List[Author] = Field(default_factory=list)
    venue: Optional[str] = None
    citations: Optional[int] = None
    references: Optional[int] = None
    url: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "PaperMetadata":
        """Create PaperMetadata from API response"""
        # Convert API author format to our Author model
        authors = [Author(**author) for author in data.get("authors", [])]

        return cls(
            paperId=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            authors=authors,
            venue=data.get("venue"),
            citations=data.get("citationCount"),  # Map from API response
            references=data.get("referenceCount"),
            url=data.get("url"),
        )


class SearchResults(BaseModel):
    """Model for search results"""

    total: int
    offset: int
    next: Optional[int] = None
    papers: List[PaperMetadata] = Field(default_factory=list)


class SemanticScholarClient:
    """Client for Semantic Scholar Academic Graph API"""

    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.headers = {"Accept": "application/json"}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    async def search_papers(
        self,
        query: str,
        offset: int = 0,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        year: Optional[int] = None,
    ) -> SearchResults:
        """Search for papers with enhanced error handling and debugging"""
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "venue",
                "citationCount",
                "referenceCount",
                "url",
            ]

        params = {
            "query": query,
            "offset": offset,
            "limit": min(limit, 100),
            "fields": ",".join(fields),
        }

        if year:
            params["year"] = str(year)

        print(
            f"Making API request to Semantic Scholar:\n{{'url': '{self.base_url}/paper/search', 'params': {params}}}"
        )

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    f"{self.base_url}/paper/search", params=params, timeout=30
                ) as response:
                    response_text = await response.text()
                    print(f"API Response Status: {response.status}")

                    if response.status != 200:
                        raise Exception(
                            f"Semantic Scholar API error ({response.status}): {response_text}"
                        )

                    data = await response.json()

                    # Transform API response into our models
                    papers = [
                        PaperMetadata.from_api_response(paper)
                        for paper in data.get("data", [])
                    ]

                    result = SearchResults(
                        total=data.get("total", 0),
                        offset=data.get("offset", 0),
                        next=data.get("next"),
                        papers=papers,
                    )

                    print(f"Successfully processed {len(papers)} papers")
                    return result

        except asyncio.TimeoutError:
            raise Exception("Request to Semantic Scholar API timed out")
        except aiohttp.ClientError as e:
            raise Exception(
                f"Network error while connecting to Semantic Scholar API: {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error processing Semantic Scholar API request: {str(e)}")

    async def get_paper_details(
        self, paper_id: str, fields: Optional[List[str]] = None
    ) -> PaperMetadata:
        """
        Get detailed information about a specific paper

        Args:
            paper_id: Semantic Scholar Paper ID
            fields: List of fields to include in response

        Returns:
            PaperMetadata object with paper details
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "venue",
                "citationCount",
                "referenceCount",
                "url",
            ]

        params = {"fields": ",".join(fields)}

        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/paper/{paper_id}", params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Semantic Scholar API error ({response.status}): {error_text}"
                    )

                data = await response.json()
                return PaperMetadata(**data)


# Helper function for synchronous calls
def sync_search_papers(
    query: str,
    offset: int = 0,
    limit: int = 10,
    fields: Optional[List[str]] = None,
    year: Optional[int] = None,
) -> SearchResults:
    """Synchronous wrapper for search_papers method"""

    async def _search():
        client = SemanticScholarClient()
        return await client.search_papers(
            query=query, offset=offset, limit=limit, fields=fields, year=year
        )

    return asyncio.run(_search())


if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = SemanticScholarClient()
        print("Testing Semantic Scholar API client...")

        try:
            # Test paper search
            results = await client.search_papers(query="large language models", limit=5)
            print(
                f"\nFound {results.total} papers. First {len(results.papers)} results:"
            )
            for paper in results.papers:
                print(f"\nTitle: {paper.title}")
                print(f"Year: {paper.year}")
                print(f"Citations: {paper.citationCount}")
                if paper.abstract:
                    print(f"Abstract: {paper.abstract[:200]}...")

            # Test paper details
            if results.papers:
                paper_id = results.papers[0].paperId
                print(f"\nGetting details for paper {paper_id}...")
                paper = await client.get_paper_details(paper_id)
                print(f"Title: {paper.title}")
                print(f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}")

        except Exception as e:
            print(f"Error during test: {str(e)}")

    asyncio.run(test_client())
