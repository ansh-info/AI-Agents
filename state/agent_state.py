from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class PaperContext(BaseModel):
    """Enhanced model for tracking paper details in conversation"""

    paper_id: str = Field(alias="paperId")
    title: str = Field(default="Untitled Paper")
    authors: List[Dict[str, Any]]
    year: Optional[int] = None
    citations: Optional[int] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    last_referenced: Optional[datetime] = Field(default_factory=datetime.now)
    reference_count: int = 0
    discussed_aspects: Set[str] = Field(default_factory=set)


class SearchContext(BaseModel):
    query: str = ""
    results: Optional[Any] = None
    current_page: int = 1
    total_results: int = 0
    selected_paper_index: Optional[int] = None
    current_filters: Dict[str, Any] = Field(default_factory=dict)  # Added this field

    class Config:
        arbitrary_types_allowed = True


class ConversationMemory(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_context: Optional[str] = None
    last_command: Optional[str] = None

    def model_copy(self, *args, **kwargs):
        return ConversationMemory(
            messages=self.messages.copy(),
            current_context=self.current_context,
            last_command=self.last_command,
        )

    class Config:
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    status: AgentStatus = AgentStatus.IDLE
    error_message: Optional[str] = None
    search_context: SearchContext = Field(default_factory=SearchContext)
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    current_step: str = "initial"
    next_steps: List[str] = Field(default_factory=list)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation memory"""
        self.memory.messages.append({"role": role, "content": content})

    def update_search_results(self, results: pd.DataFrame, total_results: int):
        """Update search results in the search context"""
        self.search_context.results = results
        self.search_context.total_results = total_results
        self.status = AgentStatus.SUCCESS

    class Config:
        arbitrary_types_allowed = True

    def update_status(self, new_status: AgentStatus, error: Optional[str] = None):
        self.status = new_status
        self.error_message = error if new_status == AgentStatus.ERROR else None

    def add_message(self, role: str, content: str):
        self.memory.messages.append({"role": role, "content": content})

    def update_search_results(self, results: pd.DataFrame, total_results: int):
        self.search_context.results = results
        self.search_context.total_results = total_results
        self.update_status(AgentStatus.SUCCESS)
