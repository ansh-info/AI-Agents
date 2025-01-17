from typing import Annotated, Any, Dict, List, Optional

from langchain_core.tools import tool

from state.agent_state import AgentState
from tools.ollama_tool import OllamaTool
from tools.semantic_scholar_tool import SemanticScholarTool
from tools.state_tool import StateTool


class ResearchTools:
    """Collection of research-related tools for the agent."""

    def __init__(self, state: AgentState = None):
        self.s2_tool = SemanticScholarTool()
        self.ollama_tool = OllamaTool()
        self.state_tool = StateTool(state) if state else None

    @tool
    async def search_papers(
        self,
        query: Annotated[str, "Search query for academic papers"],
        year_start: Annotated[Optional[int], "Start year filter"] = None,
        year_end: Annotated[Optional[int], "End year filter"] = None,
        min_citations: Annotated[Optional[int], "Minimum citations filter"] = None,
    ) -> str:
        """Search for academic papers using Semantic Scholar."""
        return await self.s2_tool.search_papers(
            query=query,
            year_start=year_start,
            year_end=year_end,
            min_citations=min_citations,
        )

    @tool
    async def get_paper_details(
        self, paper_id: Annotated[str, "The ID of the paper to retrieve details for"]
    ) -> str:
        """Get detailed information about a specific paper."""
        return await self.s2_tool.get_paper_details(paper_id)

    @tool
    async def analyze_paper(
        self,
        paper_id: Annotated[str, "ID of the paper to analyze"],
        question: Annotated[str, "Question about the paper"],
    ) -> str:
        """Analyze a paper using LLM."""
        paper_context = await self.state_tool.get_paper_context(paper_id)
        prompt = f"""Based on this paper:
{paper_context}

Question: {question}

Please provide a detailed analysis addressing the question."""

        return await self.ollama_tool.generate_response(prompt=prompt, temperature=0.7)

    @tool
    async def compare_papers(
        self, paper_ids: Annotated[List[str], "List of paper IDs to compare"]
    ) -> str:
        """Compare multiple papers."""
        papers_context = []
        for pid in paper_ids:
            context = await self.state_tool.get_paper_context(pid)
            papers_context.append(context)

        prompt = f"""Compare these papers:
{'\n---\n'.join(papers_context)}

Consider:
1. Main findings and contributions
2. Methodological approaches
3. Key differences and similarities
4. Impact and significance

Provide a structured comparison."""

        return await self.ollama_tool.generate_response(prompt=prompt, temperature=0.7)
