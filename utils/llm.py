from typing import Any, Dict

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama

from config.config import config


def create_llm() -> Ollama:
    """Create and configure Ollama LLM instance"""
    return Ollama(model=config.LLM_MODEL, temperature=config.TEMPERATURE)


class LLMManager:
    def __init__(self):
        self.llm = create_llm()

    def get_response(
        self,
        system_prompt: str,
        user_input: str,
        additional_context: Dict[str, Any] = None,
    ) -> str:
        """
        Get response from LLM with system prompt and user input

        Args:
            system_prompt: The system prompt to guide the LLM
            user_input: The user's input/query
            additional_context: Optional additional context to append to user input

        Returns:
            str: The LLM's response
        """
        # Prepare context string if provided
        context_str = ""
        if additional_context:
            context_str = "\nContext:\n" + "\n".join(
                f"{k}: {v}" for k, v in additional_context.items()
            )

        # Combine user input with context
        full_input = user_input + context_str

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_input),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"


# Create a global instance
llm_manager = LLMManager()
