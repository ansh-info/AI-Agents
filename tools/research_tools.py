from typing import Annotated, Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from state.agent_state import AgentState
from tools.ollama_tool import OllamaTool
from tools.semantic_scholar_tool import SemanticScholarTool
from tools.state_tool import StateTool


class AnalyzePaperSchema(BaseModel):
    """Schema for paper analysis parameters"""

    paper_id: str = Field(..., description="ID of the paper to analyze")
    question: str = Field(..., description="Question about the paper")


class ComparePapersSchema(BaseModel):
    """Schema for paper comparison parameters"""

    paper_ids: List[str] = Field(..., description="List of paper IDs to compare")


class ResearchTools(BaseTool):
    """Collection of research-related tools for the agent."""

    name: str = "research_tools"
    description: str = "Collection of tools for academic research"
    args_schema: Type[BaseModel] = AnalyzePaperSchema  # Default schema
    _s2_tool: SemanticScholarTool = PrivateAttr()
    _ollama_tool: OllamaTool = PrivateAttr()
    _state_tool: Optional[StateTool] = PrivateAttr()

    def __init__(self, state: AgentState = None):
        """Initialize research tools"""
        super().__init__()
        self._s2_tool = SemanticScholarTool()
        self._ollama_tool = OllamaTool()
        self._state_tool = StateTool(state) if state else None

    async def analyze_paper(self, paper_id: str, question: str) -> str:
        """Analyze a paper using LLM."""
        try:
            if not self._state_tool:
                return "State tool not initialized"

            paper_context = await self._state_tool._arun(paper_id)
            prompt = (
                f"Based on this paper:\n"
                f"{paper_context}\n\n"
                f"Question: {question}\n\n"
                f"Please provide a detailed analysis addressing the question."
            )

            return await self._ollama_tool._arun(prompt=prompt, temperature=0.7)
        except Exception as e:
            return f"Error analyzing paper: {str(e)}"

    async def compare_papers(self, paper_ids: List[str]) -> str:
        """Compare multiple papers."""
        try:
            if not self._state_tool:
                return "State tool not initialized"

            papers_context = []
            for pid in paper_ids:
                context = await self._state_tool._arun(pid)
                papers_context.append(context)

            papers_text = "\n---\n".join(papers_context)
            prompt = (
                f"Compare these papers:\n"
                f"{papers_text}\n\n"
                f"Consider:\n"
                f"1. Main findings and contributions\n"
                f"2. Methodological approaches\n"
                f"3. Key differences and similarities\n"
                f"4. Impact and significance\n\n"
                f"Provide a structured comparison."
            )

            return await self._ollama_tool._arun(prompt=prompt, temperature=0.7)
        except Exception as e:
            return f"Error comparing papers: {str(e)}"

    async def _arun(self, paper_id: str, question: str) -> str:
        """Run the default tool action (paper analysis)."""
        return await self.analyze_paper(paper_id, question)

    def _run(self, *args, **kwargs) -> str:
        """Synchronous execution not supported."""
        raise NotImplementedError("This tool only supports async execution")

    async def search_papers(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> str:
        """Search for papers using Semantic Scholar."""
        try:
            return await self._s2_tool._arun(
                query=query,
                year_start=year_start,
                year_end=year_end,
                min_citations=min_citations,
            )
        except Exception as e:
            return f"Error searching papers: {str(e)}"

    async def get_paper_details(self, paper_id: str) -> str:
        """Get detailed information about a paper."""
        try:
            if not self._state_tool:
                return "State tool not initialized"
            return await self._state_tool._arun(paper_id)
        except Exception as e:
            return f"Error getting paper details: {str(e)}"
