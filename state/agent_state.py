from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class SearchContext(BaseModel):
    query: str = ""
    results: Optional[Any] = None
    current_page: int = 1
    total_results: int = 0
    selected_paper_index: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class ConversationMemory(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_context: Optional[str] = None
    last_command: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    status: AgentStatus = AgentStatus.IDLE
    error_message: Optional[str] = None
    search_context: SearchContext = Field(default_factory=SearchContext)
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    current_step: str = "initial"
    next_steps: List[str] = Field(default_factory=list)

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
