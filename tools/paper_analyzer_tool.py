from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, PaperContext


class AnalysisRequest(BaseModel):
    """Schema for paper analysis requests"""

    paper_id: Optional[str] = Field(None, description="ID of the paper to analyze")
    paper_index: Optional[int] = Field(
        None, description="Index of the paper in search results (1-based)"
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis requested (summary, methodology, findings, impact, comparison)",
    )
    specific_question: Optional[str] = Field(
        None, description="Specific question about the paper"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context for analysis"
    )


class PaperAnalyzerTool(BaseTool):
    """Tool for in-depth analysis of research papers"""

    name: str = "paper_analyzer_tool"
    description: str = """Use this tool for detailed analysis of research papers.
    
    Capabilities:
    1. Summarize papers
    2. Analyze methodology
    3. Extract key findings
    4. Assess research impact
    5. Compare multiple papers
    6. Answer specific questions about papers
    
    Input should specify the paper (by ID or index) and type of analysis needed.
    """
    args_schema: Type[BaseModel] = AnalysisRequest

    # Private attributes
    _ollama_client: OllamaClient = PrivateAttr()
    _s2_client: SemanticScholarClient = PrivateAttr()
    _state: Optional[AgentState] = PrivateAttr()

    def __init__(
        self,
        model_name: str = "llama3.2:1b-instruct-q3_K_M",
        state: Optional[AgentState] = None,
    ):
        """Initialize the tool with required clients and state"""
        super().__init__()
        print("[DEBUG] Initializing PaperAnalyzerTool")
        self._ollama_client = OllamaClient(model_name=model_name)
        self._s2_client = SemanticScholarClient()
        self._state = state
        print("[DEBUG] PaperAnalyzerTool initialized")

    async def _arun(
        self,
        paper_id: Optional[str] = None,
        paper_index: Optional[int] = None,
        analysis_type: str = "summary",
        specific_question: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the tool asynchronously"""
        try:
            print("[DEBUG] PaperAnalyzerTool: Processing analysis request")

            # Get paper context
            paper = await self._get_paper_context(paper_id, paper_index)
            if not paper:
                return "Could not find the specified paper. Please provide a valid paper ID or index."

            # Update state if available
            if self._state:
                self._state.memory.set_focused_paper(paper)

            # Generate analysis based on type
            analysis = await self._generate_analysis(
                paper, analysis_type, specific_question, context
            )

            print(f"[DEBUG] Analysis generated, length: {len(analysis)}")
            return analysis

        except Exception as e:
            error_msg = f"Error analyzing paper: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    async def _get_paper_context(
        self, paper_id: Optional[str], paper_index: Optional[int]
    ) -> Optional[PaperContext]:
        """Retrieve paper context from state or fetch from API"""
        try:
            # First try to get from state
            if self._state:
                if paper_id:
                    paper = self._state.search_context.get_paper_by_id(paper_id)
                    if paper:
                        return paper
                elif paper_index:
                    paper = self._state.search_context.get_paper_by_index(paper_index)
                    if paper:
                        return paper

            # If not in state and we have an ID, fetch from API
            if paper_id:
                paper_data = await self._s2_client.get_paper_details(paper_id)
                if paper_data:
                    return PaperContext(**paper_data)

            return None

        except Exception as e:
            print(f"[DEBUG] Error getting paper context: {str(e)}")
            return None

    async def _generate_analysis(
        self,
        paper: PaperContext,
        analysis_type: str,
        specific_question: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate paper analysis using Ollama"""
        # Build paper context
        paper_context = f"""
Title: {paper.title}
Authors: {", ".join(a.get("name", "") for a in paper.authors)}
Year: {paper.year or "N/A"}
Citations: {paper.citations or 0}
Abstract: {paper.abstract or "Not available"}
"""

        # Select prompt based on analysis type
        prompts = {
            "summary": f"""Provide a comprehensive summary of this paper:
{paper_context}

Focus on:
1. Main research question/objective
2. Key methodology
3. Principal findings
4. Significance of results""",
            "methodology": f"""Analyze the methodology of this paper:
{paper_context}

Focus on:
1. Research approach
2. Data collection methods
3. Analysis techniques
4. Methodological strengths/limitations""",
            "findings": f"""Extract and explain the key findings of this paper:
{paper_context}

Focus on:
1. Main results
2. Supporting evidence
3. Statistical significance
4. Practical implications""",
            "impact": f"""Assess the research impact of this paper:
{paper_context}

Focus on:
1. Citation metrics
2. Field influence
3. Practical applications
4. Future research implications""",
            "comparison": f"""Compare this paper with other relevant papers in the field:
{paper_context}

Focus on:
1. Novel contributions
2. Relationship to existing work
3. Methodological differences
4. Comparative strengths/weaknesses""",
        }

        # Build final prompt
        if specific_question:
            base_prompt = f"""Based on this paper:
{paper_context}

Answer this specific question: {specific_question}

Consider:
1. Relevant aspects from the paper
2. Supporting evidence
3. Any limitations or caveats
4. Broader context"""
        else:
            base_prompt = prompts.get(
                analysis_type.lower(),
                prompts["summary"],  # Default to summary if type not recognized
            )

        # Generate analysis
        return await self._ollama_client.generate(
            prompt=base_prompt,
            system_prompt="You are an expert research paper analyst. Provide clear, accurate, and well-structured analysis.",
            max_tokens=500,
        )

    def set_state(self, state: AgentState):
        """Set or update the current state"""
        self._state = state
        print("[DEBUG] Updated PaperAnalyzerTool state")

    async def check_health(self) -> bool:
        """Check if the tool is functioning properly"""
        try:
            print("[DEBUG] Checking PaperAnalyzerTool health")
            ollama_health = await self._ollama_client.check_model_availability()
            s2_health = await self._s2_client.check_api_status()
            health_status = ollama_health and s2_health
            print(f"[DEBUG] PaperAnalyzerTool health check result: {health_status}")
            return health_status
        except Exception as e:
            print(f"[DEBUG] PaperAnalyzerTool health check failed: {str(e)}")
            return False
