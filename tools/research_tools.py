from typing import Dict, List, Optional

from langchain_core.tools import BaseTool

from state.agent_state import AgentState
from tools.ollama_tool import OllamaTool
from tools.semantic_scholar_tool import SemanticScholarTool


class ResearchTools:
    """Collection of research-related tools."""

    def __init__(self, state: Optional[AgentState] = None):
        """Initialize research tools"""
        print("[DEBUG] Initializing ResearchTools")

        # Initialize individual tools
        self._semantic_scholar = SemanticScholarTool()
        self._ollama = OllamaTool()

        # Store all tools
        self.tools: List[BaseTool] = [
            self._semantic_scholar,
            self._ollama,
        ]

        print(f"[DEBUG] Initialized {len(self.tools)} tools")
        self._state = state

    async def search_papers(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> str:
        """Search for papers using Semantic Scholar."""
        try:
            print(f"[DEBUG] ResearchTools: Searching papers with query: {query}")
            result = await self._semantic_scholar._arun(
                query=query,
                year_start=year_start,
                year_end=year_end,
                min_citations=min_citations,
            )
            print(f"[DEBUG] Search completed with result length: {len(result)}")
            return result
        except Exception as e:
            error_msg = f"Error searching papers: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama."""
        try:
            print(
                f"[DEBUG] ResearchTools: Generating text for prompt: {prompt[:100]}..."
            )
            result = await self._ollama._arun(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            print(f"[DEBUG] Text generation completed with length: {len(result)}")
            return result
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    def get_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return self.tools

    async def check_health(self) -> Dict[str, bool]:
        """Check health of all tools."""
        try:
            print("[DEBUG] Checking health of all tools")
            results = {
                "semantic_scholar": await self._semantic_scholar.check_health(),
                "ollama": await self._ollama.check_health(),
            }
            print(f"[DEBUG] Health check results: {results}")
            return results
        except Exception as e:
            print(f"[DEBUG] Health check failed: {str(e)}")
            return {"semantic_scholar": False, "ollama": False, "error": str(e)}
