from typing import Annotated, Optional

from langchain_core.tools import BaseTool, tool
from ollama_client import OllamaClient


class OllamaTool(BaseTool):
    """Tool for interacting with Ollama LLM"""

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        self.client = OllamaClient(model_name=model_name)
        super().__init__()

    @tool
    async def generate_response(
        self,
        prompt: Annotated[str, "The prompt to send to the LLM"],
        system_prompt: Annotated[Optional[str], "Optional system prompt"] = None,
        max_tokens: Annotated[Optional[int], "Maximum tokens to generate"] = None,
        temperature: Annotated[float, "Temperature for generation"] = 0.7,
    ) -> str:
        """Generate a response using the Ollama LLM."""
        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    @tool
    async def check_availability(self) -> str:
        """Check if the Ollama model is available."""
        try:
            is_available = await self.client.check_model_availability()
            return "Model is available" if is_available else "Model is not available"
        except Exception as e:
            return f"Error checking model availability: {str(e)}"
