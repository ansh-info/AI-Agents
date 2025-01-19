from typing import Annotated, Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

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
    _client: OllamaClient = PrivateAttr()  # Use PrivateAttr for non-serialized fields

    def __init__(self):
        super().__init__()
        print("[DEBUG] Initializing OllamaTool")
        self._client = OllamaClient()

    async def _arun(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        try:
            print(
                f"[DEBUG] OllamaTool: Generating response for prompt: {prompt[:100]}..."
            )
            print(
                f"[DEBUG] System prompt: {system_prompt[:100] if system_prompt else 'None'}"
            )

            response = await self._client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            print(f"[DEBUG] Generated response length: {len(response)}")
            return response

        except Exception as e:
            error_msg = f"Error in OllamaTool: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    def _run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Synchronously generate a response (not implemented)."""
        raise NotImplementedError("This tool only supports async execution")
