import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import aiohttp
from pydantic import BaseModel, Field, validator


class Author(BaseModel):
    """Enhanced model for author information"""

    authorId: Optional[str] = None
    name: str
    url: Optional[str] = None
    affiliations: List[str] = Field(default_factory=list)

    @validator("name")
    def validate_name(cls, v):
        """Ensure name is properly formatted"""
        if not v:
            return "Unknown Author"
        return v.strip()


class PaperMetadata(BaseModel):
    """Enhanced model for paper metadata"""

    paperId: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: List[Author] = Field(default_factory=list)
    venue: Optional[str] = None
    citations: Optional[int] = None
    references: Optional[int] = None
    url: Optional[str] = None
    fieldsOfStudy: List[str] = Field(default_factory=list)
    isOpenAccess: Optional[bool] = None
    tldr: Optional[str] = None  # Short summary if available

    @validator("title")
    def validate_title(cls, v):
        """Ensure title is properly formatted"""
        if not v:
            return "Untitled Paper"
        return v.strip()

    @validator("abstract")
    def validate_abstract(cls, v):
        """Clean abstract text"""
        if not v:
            return None
        return v.strip()


class SearchFilters(BaseModel):
    """Model for search filters"""

    year_start: Optional[int] = None
    year_end: Optional[int] = None
    min_citations: Optional[int] = None
    fields_of_study: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    is_open_access: Optional[bool] = None
    has_pdf: Optional[bool] = None

    def to_params(self) -> Dict[str, Any]:
        """Convert filters to API parameters"""
        params = {}
        if self.year_start:
            params["year"] = f">={self.year_start}"
        if self.year_end:
            if "year" in params:
                params["year"] = f"{params['year']},{self.year_end}"
            else:
                params["year"] = f"<={self.year_end}"
        if self.min_citations:
            params["citationCount"] = f">={self.min_citations}"
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        if self.venue:
            params["venue"] = self.venue
        if self.is_open_access is not None:
            params["isOpenAccess"] = str(self.is_open_access).lower()
        if self.has_pdf is not None:
            params["hasPdf"] = str(self.has_pdf).lower()
        return params


class SearchResults(BaseModel):
    """Enhanced model for search results"""

    total: int
    offset: int
    next: Optional[int] = None
    papers: List[PaperMetadata] = Field(default_factory=list)
    query_time: float = 0.0
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


class SemanticScholarClient:
    """Enhanced client for Semantic Scholar Academic Graph API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        max_retries: int = 3,
        timeout: int = 30,
        requests_per_minute: int = 100,
    ):
        """Initialize the client with enhanced configuration"""
        self.base_url = base_url
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout

        # Rate limiting configuration
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.min_request_interval = 60.0 / requests_per_minute

        # Initialize headers
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "AcademicSearchAssistant/1.0",
        }
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    async def _wait_for_rate_limit(self):
        """Enhanced rate limiting with better timing"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def search_papers(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        offset: int = 0,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
    ) -> SearchResults:
        """Enhanced paper search with comprehensive options"""
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
                "fieldsOfStudy",
                "isOpenAccess",
                "url",
            ]

        # Build query parameters
        params = {
            "query": query,
            "offset": offset,
            "limit": min(limit, 10),  # API limit is 100, but we're limiting to 10
            "fields": ",".join(fields),
        }

        # Add filters if provided
        if filters:
            params.update(filters.to_params())

        # Add sorting if specified
        if sort_by:
            params["sort"] = sort_by

        start_time = time.time()
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Apply rate limiting
                await self._wait_for_rate_limit()

                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get(
                        f"{self.base_url}/paper/search",
                        params=params,
                        timeout=self.timeout,
                    ) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = float(
                                response.headers.get("Retry-After", 1.0)
                            )
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue

                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(
                                f"API error ({response.status}): {error_text}"
                            )

                        data = await response.json()
                        papers = []

                        for paper_data in data.get("data", []):
                            # Convert authors data
                            authors = [
                                Author(
                                    authorId=author.get("authorId"),
                                    name=author.get("name", ""),
                                    url=author.get("url"),
                                    affiliations=author.get("affiliations", []),
                                )
                                for author in paper_data.get("authors", [])
                            ]

                            # Create paper metadata
                            paper = PaperMetadata(
                                paperId=paper_data.get("paperId", ""),
                                title=paper_data.get("title", ""),
                                abstract=paper_data.get("abstract"),
                                year=paper_data.get("year"),
                                authors=authors,
                                venue=paper_data.get("venue"),
                                citations=paper_data.get("citationCount"),
                                references=paper_data.get("referenceCount"),
                                url=paper_data.get("url"),
                                topics=paper_data.get("topics", []),
                                fieldsOfStudy=paper_data.get("fieldsOfStudy", []),
                                isOpenAccess=paper_data.get("isOpenAccess"),
                                tldr=paper_data.get("tldr", {}).get("text"),
                            )
                            papers.append(paper)

                        # Calculate query time
                        query_time = time.time() - start_time

                        return SearchResults(
                            total=data.get("total", 0),
                            offset=data.get("offset", 0),
                            next=data.get("next"),
                            papers=papers,
                            query_time=query_time,
                            filters_applied=filters.dict() if filters else {},
                        )

            except aiohttp.ClientError as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise Exception(
                        f"Network error after {retry_count} attempts: {str(e)}"
                    )
                await asyncio.sleep(1.0 * retry_count)

            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise Exception(f"Error after {retry_count} attempts: {str(e)}")
                await asyncio.sleep(1.0 * retry_count)

        raise Exception(f"Failed to get response after {self.max_retries} attempts")

    async def get_paper_details(self, paper_id: str) -> PaperMetadata:
        """Get detailed information about a specific paper"""
        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "venue",
            "citationCount",
            "referenceCount",
            "fieldsOfStudy",
            "topics",
            "isOpenAccess",
            "tldr",
            "url",
        ]

        try:
            await self._wait_for_rate_limit()

            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    f"{self.base_url}/paper/{paper_id}",
                    params={"fields": ",".join(fields)},
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API error ({response.status})")

                    paper_data = await response.json()

                    # Convert authors
                    authors = [
                        Author(
                            authorId=author.get("authorId"),
                            name=author.get("name", ""),
                            url=author.get("url"),
                            affiliations=author.get("affiliations", []),
                        )
                        for author in paper_data.get("authors", [])
                    ]

                    return PaperMetadata(
                        paperId=paper_data.get("paperId", ""),
                        title=paper_data.get("title", ""),
                        abstract=paper_data.get("abstract"),
                        year=paper_data.get("year"),
                        authors=authors,
                        venue=paper_data.get("venue"),
                        citations=paper_data.get("citationCount"),
                        references=paper_data.get("referenceCount"),
                        url=paper_data.get("url"),
                        topics=paper_data.get("topics", []),
                        fieldsOfStudy=paper_data.get("fieldsOfStudy", []),
                        isOpenAccess=paper_data.get("isOpenAccess"),
                        tldr=paper_data.get("tldr", {}).get("text"),
                    )

        except Exception as e:
            raise Exception(f"Error getting paper details: {str(e)}")

    async def check_api_status(self) -> bool:
        """Check if the API is available and responding"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    f"{self.base_url}/paper/search",
                    params={"query": "test", "limit": 1},
                    timeout=5,
                ) as response:
                    return response.status == 200
        except:
            return False
