from typing import Annotated, Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from clients.ollama_client import OllamaClient


class GenerateResponseSchema(BaseModel):
    """Schema for generate_response parameters"""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")


class OllamaTool(BaseTool):
    """Tool for interacting with Ollama LLM"""

    name: str = "ollama_tool"
    description: str = "Tool for interacting with the Ollama LLM"
    args_schema: Type[BaseModel] = GenerateResponseSchema

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize the tool with a specific model"""
        super().__init__()
        self.client = OllamaClient(model_name=model_name)

    async def _arun(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Asynchronously generate a response using the Ollama LLM."""
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

    def _run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Synchronously generate a response (not implemented)."""
        raise NotImplementedError("This tool only supports async execution")
