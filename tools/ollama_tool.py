import asyncio
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from clients.ollama_client import OllamaClient
from state.agent_state import AgentState


class GenerateInput(BaseModel):
    """Schema for text generation parameters"""

    prompt: str = Field(..., description="The prompt to send to the model")
    system_prompt: Optional[str] = Field(
        None, description="Optional system prompt to guide the model's behavior"
    )
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(
        0.7, description="Temperature for generation (0.0 to 1.0)"
    )


class OllamaTool(BaseTool):
    """Tool for LLM interactions using Ollama"""

    name: str = "ollama_tool"
    description: str = """Use this tool for text generation and understanding.
    
    Capabilities:
    1. Generate responses to questions
    2. Analyze and understand user intents
    3. Provide explanations and summaries
    4. Help with paper analysis
    """
    args_schema: Type[BaseModel] = GenerateInput

    # Private attributes
    _client: OllamaClient = PrivateAttr()
    _state: Optional[AgentState] = PrivateAttr()

    def __init__(
        self,
        model_name: str = "llama3.2:1b-instruct-q3_K_M",
        state: Optional[AgentState] = None,
    ):
        """Initialize the tool with a specific model and optional state"""
        super().__init__()
        print("[DEBUG] Initializing OllamaTool")
        self._client = OllamaClient(model_name=model_name)
        self._state = state
        print(f"[DEBUG] OllamaTool initialized with model: {model_name}")

    async def _arun(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Run the tool asynchronously with enhanced error handling"""
        try:
            print(
                f"[DEBUG] OllamaTool: Generating response for prompt: {prompt[:100]}..."
            )

            # Build context from state using existing method
            context = self._build_context()

            # Combine context and prompt
            full_prompt = f"{context}\n\nUser request: {prompt}" if context else prompt

            # Generate response with retries
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = await self._client.generate(
                        prompt=full_prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    # Update state if available
                    if self._state:
                        self._state.add_message("user", prompt)
                        self._state.add_message("system", response)

                    return response

                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:  # Not the last attempt
                        print(f"[DEBUG] Retry {attempt + 1} after error: {str(e)}")
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                    break  # On last attempt, break and handle error

            # If we get here, all retries failed
            error_msg = (
                f"Error in OllamaTool after {max_retries} attempts: {str(last_error)}"
            )
            print(f"[DEBUG] {error_msg}")
            if self._state:
                self._state.add_message("system", error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Unexpected error in OllamaTool: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            if self._state:
                self._state.add_message("system", error_msg)
            return error_msg

    def _run(self, prompt: str, **kwargs) -> str:
        """Synchronous execution not supported"""
        raise NotImplementedError("This tool only supports async execution")

    def set_state(self, state: AgentState):
        """Set or update the current state"""
        self._state = state
        print("[DEBUG] Updated OllamaTool state")

    def _build_context(self) -> str:
        """Build context from current state"""
        if not self._state:
            return ""

        context_parts = []

        # Add recent conversation history
        if self._state.memory and self._state.memory.messages:
            recent_messages = self._state.memory.messages[-5:]  # Last 5 messages
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # Add current paper context if available
        if self._state.memory and self._state.memory.focused_paper:
            paper = self._state.memory.focused_paper
            context_parts.extend(
                [
                    "\nCurrently discussing paper:",
                    f"Title: {paper.title}",
                    f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}",
                    f"Year: {paper.year or 'N/A'}",
                    f"Citations: {paper.citations or 0}",
                    f"Abstract: {paper.abstract[:200] + '...' if paper.abstract else 'Not available'}",
                ]
            )

        return "\n".join(context_parts)

    def _update_state(self, prompt: str, response: str):
        """Update state with new interaction"""
        if not self._state:
            return

        # Add messages to state
        self._state.add_message("user", prompt)
        self._state.add_message("system", response)

    async def check_health(self) -> bool:
        """Check if the model is available and responding"""
        try:
            print("[DEBUG] Checking OllamaTool health")
            is_available = await self._client.check_model_availability()
            print(f"[DEBUG] OllamaTool health check result: {is_available}")
            return is_available
        except Exception as e:
            print(f"[DEBUG] OllamaTool health check failed: {str(e)}")
            return False
