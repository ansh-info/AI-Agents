from typing import Any, Dict

from config.config import config


class SharedState:
    def __init__(self):
        self.state: Dict[str, Any] = {
            config.StateKeys.PAPERS: [],
            config.StateKeys.SELECTED_PAPERS: [],
            config.StateKeys.CURRENT_TOOL: None,
            config.StateKeys.CURRENT_AGENT: None,
            config.StateKeys.RESPONSE: None,
            config.StateKeys.ERROR: None,
        }

    def get(self, key: str) -> Any:
        """Get value from state"""
        return self.state.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in state"""
        self.state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in state"""
        self.state.update(updates)

    def clear_response(self) -> None:
        """Clear response and error fields"""
        self.state[config.StateKeys.RESPONSE] = None
        self.state[config.StateKeys.ERROR] = None

    def add_papers(self, papers: list) -> None:
        """Add papers to the papers list"""
        self.state[config.StateKeys.PAPERS].extend(papers)

    def select_papers(self, paper_ids: list) -> None:
        """Select papers from the papers list"""
        selected = [
            paper
            for paper in self.state[config.StateKeys.PAPERS]
            if paper.get("paperId") in paper_ids
        ]
        self.state[config.StateKeys.SELECTED_PAPERS] = selected

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context information"""
        return {
            "current_agent": self.state[config.StateKeys.CURRENT_AGENT],
            "current_tool": self.state[config.StateKeys.CURRENT_TOOL],
            "papers_count": len(self.state[config.StateKeys.PAPERS]),
            "selected_papers_count": len(self.state[config.StateKeys.SELECTED_PAPERS]),
        }


# Create a global instance
shared_state = SharedState()
