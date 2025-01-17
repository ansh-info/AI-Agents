from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import BaseTool, tool

from state.agent_state import AgentState, PaperContext  # Updated import path


class StateTool(BaseTool):
    """Tool for managing agent state operations"""

    def __init__(self, state: AgentState):
        self.state = state
        super().__init__()

    @tool
    def get_paper_context(
        self, paper_id: Annotated[str, "ID of the paper to retrieve from state"]
    ) -> str:
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

    @tool
    def update_paper_reference(
        self,
        paper_id: Annotated[str, "ID of the paper to update"],
        reference_count: Annotated[Optional[int], "New reference count"] = None,
        discussed_aspects: Annotated[
            Optional[list], "List of discussed aspects"
        ] = None,
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

    @tool
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

    @tool
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
