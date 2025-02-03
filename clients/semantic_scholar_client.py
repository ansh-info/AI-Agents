import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    citations: Optional[int] = Field(default=0, alias="citationCount")
    references: Optional[int] = Field(default=0, alias="referenceCount")
    url: Optional[str] = None
    fieldsOfStudy: Optional[List[str]] = Field(default_factory=list)  # Made Optional
    isOpenAccess: Optional[bool] = None

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

    @validator("fieldsOfStudy", pre=True)
    def validate_fields_of_study(cls, v):
        """Handle None value for fieldsOfStudy"""
        if v is None:
            return []
        return v

    class Config:
        populate_by_name = True


class SearchFilters(BaseModel):
    """Model for search filters with proper formatting"""

    year_start: Optional[int] = None
    year_end: Optional[int] = None
    min_citations: Optional[int] = None
    fields_of_study: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    is_open_access: Optional[bool] = None
    has_pdf: Optional[bool] = None

    def _format_year_param(self) -> Optional[str]:
        """Format year parameter according to API requirements"""
        if self.year_start and self.year_end:
            return f"{self.year_start}-{self.year_end}"
        elif self.year_start:
            return f"{self.year_start}-{datetime.now().year}"
        elif self.year_end:
            return f"1900-{self.year_end}"
        return None

    def to_params(self) -> Dict[str, Any]:
        """Convert filters to API parameters with proper formatting"""
        params = {}

        # Handle year parameter
        year_param = self._format_year_param()
        if year_param:
            params["year"] = year_param

        # Handle citation count
        if self.min_citations:
            params["citationCount"] = f">={self.min_citations}"

        # Handle other parameters
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
    """Enhanced client for Semantic Scholar Academic Graph API with rate limiting"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        requests_per_minute: int = 100,
        max_retries: int = 3,
    ):
        """Initialize the client with enhanced rate limiting"""
        self.base_url = base_url
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        # Rate limiting configuration
        self.requests_per_minute = requests_per_minute
        self.request_window = 60  # seconds
        self.request_timestamps = []
        self.max_retries = max_retries

        # Initialize headers
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "AcademicSearchAssistant/1.0",
        }
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        # Remove timestamps older than our window
        self.request_timestamps = [
            ts
            for ts in self.request_timestamps
            if ts > current_time - self.request_window
        ]
        return len(self.request_timestamps) < self.requests_per_minute

    async def _wait_for_rate_limit(self):
        """Wait until we're allowed to make another request"""
        while not await self._check_rate_limit():
            await asyncio.sleep(1)
        self.request_timestamps.append(time.time())

    async def search_papers(
        self, query: str, filters: Optional[SearchFilters] = None, limit: int = 10
    ) -> SearchResults:
        """Perform paper search with enhanced rate limiting and retries"""
        try:
            # Clean query
            clean_query = query.replace("?", "").strip()
            if not clean_query:
                raise ValueError("Empty search query")

            # Build parameters
            params = {
                "query": clean_query,
                "offset": 0,
                "limit": min(limit, 10),  # Ensure limit doesn't exceed 10
                "fields": "paperId,title,abstract,year,authors,citationCount,url",
            }

            # Add filters if provided
            if filters:
                params.update(filters.to_params())

            print(f"[DEBUG] Making request with params: {params}")

            # Implement retry logic with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    # Wait for rate limit
                    await self._wait_for_rate_limit()

                    async with aiohttp.ClientSession(headers=self.headers) as session:
                        async with session.get(
                            f"{self.base_url}/paper/search", params=params, timeout=30
                        ) as response:
                            if response.status == 429:
                                wait_time = 2**attempt  # Exponential backoff
                                print(
                                    f"[DEBUG] Rate limited, waiting {wait_time} seconds..."
                                )
                                await asyncio.sleep(wait_time)
                                continue

                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(
                                    f"API error ({response.status}): {error_text}"
                                )

                            data = await response.json()
                            return self._process_search_results(data)

                except aiohttp.ClientError as e:
                    if attempt == self.max_retries - 1:
                        raise Exception(
                            f"Network error after {self.max_retries} retries: {str(e)}"
                        )
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

            raise Exception("Max retries exceeded")

        except Exception as e:
            print(f"Search error: {str(e)}")
            raise

    def _process_search_results(self, data: dict) -> SearchResults:
        """Process search results with error handling"""
        if not isinstance(data, dict):
            raise ValueError(f"Invalid response format: {data}")

        papers = []
        for paper_data in data.get("data", []):
            if not paper_data:
                continue

            # Safely extract author information
            authors = []
            for author in paper_data.get("authors", []):
                if author and isinstance(author, dict):
                    authors.append(
                        Author(
                            authorId=author.get("authorId"),
                            name=author.get("name", "Unknown Author"),
                            url=author.get("url"),
                            affiliations=author.get("affiliations", []),
                        )
                    )

            # Create paper metadata with safe defaults
            paper = PaperMetadata(
                paperId=paper_data.get("paperId", ""),
                title=paper_data.get("title", "Untitled Paper"),
                abstract=paper_data.get("abstract"),
                year=paper_data.get("year"),
                authors=authors,
                citations=paper_data.get("citationCount", 0),
                references=paper_data.get("referenceCount", 0),
                url=paper_data.get("url"),
            )
            papers.append(paper)

        return SearchResults(
            total=data.get("total", 0),
            offset=data.get("offset", 0),
            papers=papers,
            query_time=time.time(),
        )

    async def check_api_status(self) -> bool:
        """Check if the API is available and responding"""
        try:
            await self._wait_for_rate_limit()
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    f"{self.base_url}/paper/search",
                    params={"query": "test", "limit": 1},
                    timeout=5,
                ) as response:
                    return response.status == 200
        except:
            return False
