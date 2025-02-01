from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class PaperContext(BaseModel):
    """Enhanced model for tracking paper details in conversation"""

    paper_id: str = Field(alias="paperId")
    title: str = Field(default="Untitled Paper")
    authors: List[Dict[str, str]] = Field(default_factory=list)
    year: Optional[int] = None
    citations: Optional[int] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    last_referenced: Optional[datetime] = Field(default_factory=datetime.now)
    reference_count: int = 0
    discussed_aspects: Set[str] = Field(default_factory=set)

    @validator("paper_id", pre=True)
    def validate_paper_id(cls, v):
        """Ensure paper_id is never null and is a valid string"""
        if not v:
            raise ValueError("paper_id cannot be null or empty")
        return str(v)

    @validator("authors", pre=True)
    def validate_authors(cls, v):
        """Ensure authors list is never null"""
        if not v:
            return [{"name": "Unknown Author"}]
        if isinstance(v, list):
            return [{"name": a["name"] if isinstance(a, dict) else str(a)} for a in v]
        return [{"name": str(v)}]

    def update_reference(self):
        """Update reference tracking"""
        self.last_referenced = datetime.now()
        self.reference_count += 1

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True


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
    active_papers: List[str] = Field(default_factory=list)

    def add_paper(self, paper: Dict[str, Any]):
        """Add a paper to results with enhanced tracking"""
        try:
            print("\n[DEBUG] Adding paper to SearchContext:")
            print(f"[DEBUG] Input paper data: {paper}")
            print(
                f"[DEBUG] PaperId from input: {paper.get('paperId', 'NO_ID_IN_DICT')}"
            )

            paper_ctx = PaperContext(
                paperId=paper.get("paperId"),  # Using the aliased field name
                title=paper.get("title", "Untitled Paper"),
                authors=paper.get("authors", []),
                year=paper.get("year"),
                citations=paper.get("citations", 0),
                abstract=paper.get("abstract"),
                url=paper.get("url"),
            )
            print(f"[DEBUG] Created PaperContext with ID: {paper_ctx.paper_id}")

            self.results.append(paper_ctx)

            # Track this paper as active
            if paper_ctx.paper_id not in self.active_papers:
                self.active_papers.append(paper_ctx.paper_id)

            print("[DEBUG] Successfully added paper to results")

        except Exception as e:
            print(f"[DEBUG] Error in add_paper: {str(e)}")
            raise  # Re-raise the exception after logging

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
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_topics: Set[str] = Field(default_factory=set)
    referenced_papers: Dict[str, int] = Field(default_factory=dict)

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

        # Add message
        self.messages.append(message)

        # Update context history
        context_entry = {
            "timestamp": timestamp,
            "message_type": role,
            "focused_paper": (
                self.focused_paper.paper_id if self.focused_paper else None
            ),
            "context_data": context or {},
        }
        self.context_history.append(context_entry)

        print(f"[DEBUG] Added message - Role: {role}, Content: {content[:100]}...")

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
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_update: Optional[datetime] = Field(default_factory=datetime.now)

    def add_message(
        self, role: str, content: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enhanced message addition with context tracking"""
        try:
            # Ensure memory exists
            if not hasattr(self, "memory"):
                self.memory = ConversationMemory()

            timestamp = datetime.now()

            # Create message with metadata
            message = {
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "context": context or {},
                "state_snapshot": {
                    "current_step": self.current_step,
                    "status": self.status.value,
                    "has_search_results": bool(self.search_context.results),
                    "focused_paper": self.memory.focused_paper.paper_id
                    if self.memory.focused_paper
                    else None,
                },
            }

            # Add message to memory
            self.memory.messages.append(message)

            # Track search queries
            if role == "user":
                self._track_search_query(content)

            # Update state history
            self._track_state_change(
                "message_added", {"role": role, "timestamp": timestamp}
            )

            # Update last interaction time
            self.last_update = timestamp

        except Exception as e:
            print(f"[DEBUG] Error adding message to state: {str(e)}")
            raise

    def _track_search_query(self, content: str) -> None:
        """Track search queries for history"""
        try:
            content_lower = content.lower()
            if any(
                term in content_lower
                for term in ["find", "search", "papers about", "papers by"]
            ):
                if not hasattr(self.search_context, "search_history"):
                    self.search_context.search_history = []

                self.search_context.search_history.append(
                    {
                        "timestamp": datetime.now(),
                        "query": content,
                        "results_count": len(self.search_context.results)
                        if self.search_context.results
                        else 0,
                    }
                )
                self.search_context.last_search_time = datetime.now()

        except Exception as e:
            print(f"[DEBUG] Error tracking search query: {str(e)}")

    def _track_state_change(self, change_type: str, metadata: Dict[str, Any]) -> None:
        """Enhanced state change tracking"""
        try:
            if not hasattr(self, "state_history"):
                self.state_history = []

            state_change = {
                "timestamp": datetime.now(),
                "change_type": change_type,
                "current_step": self.current_step,
                "status": self.status.value,
                "metadata": metadata,
                "focused_paper": (
                    self.memory.focused_paper.paper_id
                    if self.memory.focused_paper
                    else None
                ),
                "search_context": {
                    "has_results": bool(self.search_context.results),
                    "results_count": len(self.search_context.results)
                    if self.search_context.results
                    else 0,
                    "last_search_time": self.search_context.last_search_time
                    if hasattr(self.search_context, "last_search_time")
                    else None,
                },
            }

            self.state_history.append(state_change)

        except Exception as e:
            print(f"[DEBUG] Error tracking state change: {str(e)}")

    def get_conversation_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        if not self.memory or not self.memory.messages:
            return []
        return self.memory.messages[-limit:]

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history"""
        if not hasattr(self.search_context, "search_history"):
            return []
        return self.search_context.search_history

    def get_focused_paper_history(self) -> List[Dict[str, Any]]:
        """Get history of focused papers"""
        if not hasattr(self.memory, "paper_history"):
            self.memory.paper_history = []
        return self.memory.paper_history

    def update_state(self, **kwargs) -> None:
        """Update state fields with tracking"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.last_update = datetime.now()
        self._track_state_change("state_updated", kwargs)

    def clear_state(self, preserve_history: bool = True) -> None:
        """Clear state with option to preserve history"""
        final_state = {
            "timestamp": datetime.now(),
            "change_type": "state_cleared",
            "final_status": self.status,
            "conversation_length": len(self.memory.messages)
            if self.memory.messages
            else 0,
        }

        # Store history if requested
        old_history = self.state_history if preserve_history else []

        # Reset state
        self.status = AgentStatus.IDLE
        self.error_message = None
        self.search_context = SearchContext()
        self.memory = ConversationMemory()
        self.current_step = "initial"
        self.next_steps = []
        self.last_update = datetime.now()

        # Restore history if preserved
        if preserve_history:
            self.state_history = old_history + [final_state]
        else:
            self.state_history = [final_state]

    class Config:
        arbitrary_types_allowed = True
