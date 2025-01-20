from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.ollama_client import OllamaClient


class GenerateInput(BaseModel):
    """Schema for text generation parameters"""

    prompt: str = Field(..., description="The prompt to send to the model")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")


class OllamaTool(BaseTool):
    """Tool for Ollama LLM interactions"""

    name: str = "ollama_tool"
    description: str = """Use this tool for text generation and understanding.
    Capabilities:
    1. Generate responses to questions
    2. Analyze and understand user intents
    3. Provide explanations and summaries
    """
    args_schema: Type[BaseModel] = GenerateInput
    _client: OllamaClient = PrivateAttr()

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize the tool with a specific model"""
        super().__init__()
        print("[DEBUG] Initializing OllamaTool")
        self._client = OllamaClient(model_name=model_name)
        print(f"[DEBUG] OllamaTool initialized with model: {model_name}")

    async def _arun(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Asynchronously generate text using Ollama."""
        try:
            print(
                f"[DEBUG] OllamaTool: Generating response for prompt: {prompt[:100]}..."
            )
            if system_prompt:
                print(f"[DEBUG] With system prompt: {system_prompt[:100]}...")

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
        """Synchronous execution not supported."""
        raise NotImplementedError("This tool only supports async execution")

    async def check_health(self) -> bool:
        """Check if the model is available and responding."""
        try:
            print("[DEBUG] Checking OllamaTool health")
            is_available = await self._client.check_model_availability()
            print(f"[DEBUG] OllamaTool health check result: {is_available}")
            return is_available
        except Exception as e:
            print(f"[DEBUG] OllamaTool health check failed: {str(e)}")
            return False
