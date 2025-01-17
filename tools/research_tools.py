from typing import Annotated, Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from state.agent_state import AgentState, PaperContext
from tools.ollama_tool import OllamaTool
from tools.semantic_scholar_tool import SemanticScholarTool
from tools.state_tool import StateTool


class GetPaperContextSchema(BaseModel):
    """Schema for get_paper_context parameters"""

    paper_id: str = Field(..., description="ID of the paper to retrieve from state")


class UpdatePaperReferenceSchema(BaseModel):
    """Schema for update_paper_reference parameters"""

    paper_id: str = Field(..., description="ID of the paper to update")
    reference_count: Optional[int] = Field(None, description="New reference count")
    discussed_aspects: Optional[list] = Field(
        None, description="List of discussed aspects"
    )


class StateTool(BaseTool):
    """Tool for managing agent state operations"""

    name: str = "state_tool"
    description: str = "Tool for managing and accessing agent state"
    args_schema: Type[BaseModel] = GetPaperContextSchema

    def __init__(self, state: AgentState):
        super().__init__()
        self.state = state

    async def _arun(
        self,
        paper_id: str,
        reference_count: Optional[int] = None,
        discussed_aspects: Optional[list] = None,
    ) -> str:
        """Get paper context from state asynchronously."""
        return self._get_paper_context(paper_id)

    def _run(
        self,
        paper_id: str,
        reference_count: Optional[int] = None,
        discussed_aspects: Optional[list] = None,
    ) -> str:
        """Get paper context from state synchronously."""
        return self._get_paper_context(paper_id)

    def _get_paper_context(self, paper_id: str) -> str:
        """Get paper context from state."""
        try:
            paper = self.state.search_context.get_paper_by_id(paper_id)
            if paper:
                return f"""
Title: {paper.title}
Authors: {', '.join(a.get('name', '') for a in paper.authors)}
Year: {paper.year or 'N/A'}
Citations: {paper.citations or 0}
Abstract: {paper.abstract or 'No abstract available'}
"""
            return "Paper not found in current context"
        except Exception as e:
            return f"Error retrieving paper context: {str(e)}"

    def update_paper_reference(
        self,
        paper_id: str,
        reference_count: Optional[int] = None,
        discussed_aspects: Optional[list] = None,
    ) -> str:
        """Update paper reference information in state."""
        try:
            paper = self.state.search_context.get_paper_by_id(paper_id)
            if paper:
                if reference_count is not None:
                    paper.reference_count = reference_count
                if discussed_aspects:
                    paper.discussed_aspects.update(discussed_aspects)
                return "Paper reference updated successfully"
            return "Paper not found in current context"
        except Exception as e:
            return f"Error updating paper reference: {str(e)}"

    def get_conversation_context(self) -> str:
        """Get current conversation context."""
        try:
            messages = (
                self.state.memory.messages[-5:] if self.state.memory.messages else []
            )
            context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )
            return f"Recent conversation context:\n{context}"
        except Exception as e:
            return f"Error retrieving conversation context: {str(e)}"

    def get_state_summary(self) -> str:
        """Get a summary of current state."""
        try:
            return f"""
State Summary:
Status: {self.state.status}
Current Step: {self.state.current_step}
Error (if any): {self.state.error_message or 'None'}
Search Results: {len(self.state.search_context.results) if self.state.search_context else 0}
Messages: {len(self.state.memory.messages) if self.state.memory else 0}
"""
        except Exception as e:
            return f"Error getting state summary: {str(e)}"
