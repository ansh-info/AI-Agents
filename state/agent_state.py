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
    ):
        """Add message with state tracking"""
        # Add message directly to memory
        self.memory.add_message(role, content, context)
        self._track_state_change("message_added")
        # Ensure state fields are updated
        self.last_update = datetime.now()

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

    def update_state(self, **kwargs):
        """Update state fields safely"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now()
        self._track_state_change("state_updated")

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
