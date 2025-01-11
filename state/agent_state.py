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

    paper_id: str
    title: str
    authors: List[Dict[str, Any]]
    year: Optional[int] = None
    citations: Optional[int] = None
    abstract: Optional[str] = None
    url: Optional[str] = None

    # Added fields for better context tracking
    last_referenced: Optional[datetime] = None
    reference_count: int = 0
    discussed_aspects: Set[str] = Field(default_factory=set)

    def update_reference(self):
        """Update paper reference tracking"""
        self.last_referenced = datetime.now()
        self.reference_count += 1

    class Config:
        arbitrary_types_allowed = True

    @validator("paper_id")
    def validate_paper_id(cls, v):
        if not v:
            raise ValueError("paper_id cannot be empty")
        return v

    @validator("title")
    def validate_title(cls, v):
        if not v:
            return "Untitled Paper"
        return v.strip()

    @validator("authors")
    def validate_authors(cls, v):
        if not v:
            return [{"name": "Unknown Author", "authorId": None}]
        return v


class SearchContext(BaseModel):
    """Enhanced context for search operations"""

    query: str = ""
    results: List[PaperContext] = Field(default_factory=list)
    current_page: int = 1
    total_results: int = 0
    selected_paper_index: Optional[int] = None
    current_filters: Dict[str, Any] = Field(default_factory=dict)

    # Added fields for better search context
    search_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_search_time: Optional[datetime] = None
    active_papers: List[str] = Field(default_factory=list)  # Currently discussed papers

    def add_paper(self, paper: Dict[str, Any]):
        """Add a paper to results with enhanced tracking"""
        paper_ctx = PaperContext(
            paper_id=paper.get("paperId"),
            title=paper.get("title"),
            authors=paper.get("authors", []),
            year=paper.get("year"),
            citations=paper.get("citationCount"),
            abstract=paper.get("abstract"),
            url=paper.get("url"),
        )
        self.results.append(paper_ctx)

        # Track this paper as active
        if paper_ctx.paper_id not in self.active_papers:
            self.active_papers.append(paper_ctx.paper_id)

    def get_paper_by_index(self, index: int) -> Optional[PaperContext]:
        """Get paper by its display index (1-based) with reference update"""
        try:
            paper = self.results[index - 1]
            if paper:
                paper.update_reference()
            return paper
        except IndexError:
            return None

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperContext]:
        """Get paper by its ID with reference update"""
        for paper in self.results:
            if paper.paper_id == paper_id:
                paper.update_reference()
                return paper
        return None

    def update_search_history(self, query: str, filters: Optional[Dict] = None):
        """Track search history"""
        self.search_history.append(
            {
                "timestamp": datetime.now(),
                "query": query,
                "filters": filters or {},
                "results_count": len(self.results),
            }
        )
        self.last_search_time = datetime.now()


class ConversationMemory(BaseModel):
    """Enhanced memory for conversation tracking"""

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_context: Optional[str] = None
    focused_paper: Optional[PaperContext] = None
    paper_references: Dict[str, PaperContext] = Field(default_factory=dict)
    last_command: Optional[str] = None

    # Enhanced context tracking
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_topics: Set[str] = Field(default_factory=set)
    referenced_papers: Dict[str, int] = Field(
        default_factory=dict
    )  # Track paper reference counts

    def add_message(
        self, role: str, content: str, context: Optional[Dict[str, Any]] = None
    ):
        """Add a message with enhanced context tracking"""
        timestamp = datetime.now()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "context": context or {},
        }

        # Update context history
        context_entry = {
            "timestamp": timestamp,
            "message_type": role,
            "focused_paper": (
                self.focused_paper.paper_id if self.focused_paper else None
            ),
            "context_data": context or {},
        }

        self.messages.append(message)
        self.context_history.append(context_entry)

    def set_focused_paper(self, paper: PaperContext):
        """Set focused paper with enhanced tracking"""
        self.focused_paper = paper
        self.paper_references[paper.paper_id] = paper
        paper.update_reference()

        # Update reference counts
        self.referenced_papers[paper.paper_id] = (
            self.referenced_papers.get(paper.paper_id, 0) + 1
        )

    def get_context_window(self, size: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context with enhanced metadata"""
        recent_messages = self.messages[-size:] if self.messages else []
        context_window = []

        for msg in recent_messages:
            context_msg = msg.copy()
            if msg.get("context"):
                context_msg["context"]["focused_paper_id"] = (
                    self.focused_paper.paper_id if self.focused_paper else None
                )
            context_window.append(context_msg)

        return context_window

    def get_paper_discussion_history(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get history of discussions about a specific paper"""
        paper_messages = []
        for msg in self.messages:
            if msg.get("context") and msg["context"].get("paper_id") == paper_id:
                paper_messages.append(msg)
        return paper_messages


class AgentState(BaseModel):
    """Enhanced main state management class"""

    status: AgentStatus = AgentStatus.IDLE
    error_message: Optional[str] = None
    search_context: SearchContext = Field(default_factory=SearchContext)
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    current_step: str = "initial"
    next_steps: List[str] = Field(default_factory=list)

    # Enhanced state tracking
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_update: Optional[datetime] = None

    def add_message(
        self, role: str, content: str, context: Optional[Dict[str, Any]] = None
    ):
        """Add message with state tracking"""
        self.memory.add_message(role, content, context)
        self._track_state_change("message_added")

    def update_search_results(self, papers: List[Dict[str, Any]], total_results: int):
        """Update search results with state tracking"""
        self.search_context.results = []
        for paper in papers:
            self.search_context.add_paper(paper)
        self.search_context.total_results = total_results
        self.status = AgentStatus.SUCCESS
        self._track_state_change("search_updated")

    def focus_on_paper(self, paper_id: str) -> Optional[PaperContext]:
        """Focus conversation on specific paper with state tracking"""
        paper = self.search_context.get_paper_by_id(paper_id)
        if paper:
            self.memory.set_focused_paper(paper)
            self._track_state_change("paper_focused")
        return paper

    def get_referenced_paper(self, message: str) -> Optional[PaperContext]:
        """Enhanced paper reference extraction from message"""
        # Check numeric references
        words = message.lower().split()
        for i, word in enumerate(words):
            if word.isdigit():
                try:
                    paper_num = int(word)
                    return self.search_context.get_paper_by_index(paper_num)
                except ValueError:
                    continue
            elif word in ["paper", "study", "article"] and i > 0:
                try:
                    prev_word = words[i - 1]
                    if prev_word.isdigit():
                        paper_num = int(prev_word)
                        return self.search_context.get_paper_by_index(paper_num)
                except ValueError:
                    continue

        # Check title references
        for paper in self.search_context.results:
            if paper.title.lower() in message.lower():
                return paper

        return None

    def _track_state_change(self, change_type: str):
        """Track state changes"""
        state_change = {
            "timestamp": datetime.now(),
            "change_type": change_type,
            "status": self.status,
            "current_step": self.current_step,
            "focused_paper": (
                self.memory.focused_paper.paper_id
                if self.memory.focused_paper
                else None
            ),
        }
        self.state_history.append(state_change)
        self.last_update = datetime.now()

    def clear(self):
        """Enhanced state reset with history preservation"""
        # Store final state before clearing
        final_state = {
            "timestamp": datetime.now(),
            "change_type": "state_cleared",
            "final_status": self.status,
            "conversation_length": len(self.memory.messages),
        }
        self.state_history.append(final_state)

        # Reset state
        self.status = AgentStatus.IDLE
        self.error_message = None
        self.search_context = SearchContext()
        self.memory = ConversationMemory()
        self.current_step = "initial"
        self.next_steps = []
        self.last_update = datetime.now()

    class Config:
        arbitrary_types_allowed = True
