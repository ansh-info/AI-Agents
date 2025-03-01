from typing import Any, Dict, List, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config.config import config
from state.shared_state import shared_state


def create_llm() -> ChatOllama:
    """Create and configure Ollama LLM instance with tool calling support"""
    return ChatOllama(
        model=config.LLM_MODEL,
        format="json",  # Enable JSON mode for tool calls
        options={
            "temperature": 0.1,  # Lower temperature for more deterministic tool calling
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_predict": 1024,  # Increased context for complex tool calls
        },
    )


class LLMManager:
    def __init__(self):
        self.llm = create_llm()

    def get_response(
        self,
        system_prompt: str,
        user_input: str,
        additional_context: Dict[str, Any] = None,
        include_history: bool = True,
    ) -> str:
        """Get response from LLM with system prompt and user input"""
        try:
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]

            # Add debug logging
            print("\nDebug - LLM Input:")
            print(f"System prompt: {system_prompt[:200]}...")
            print(f"User input: {user_input}")

            # Get response with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(messages)

                    # Add debug logging
                    print(f"\nDebug - LLM Response (Attempt {attempt + 1}):")
                    print(f"Raw response: {response.content}")

                    if response and response.content.strip():
                        # Return the cleaned content without validation
                        # Let the calling function handle JSON validation
                        return response.content.strip()

                    if attempt < max_retries - 1:
                        print(f"Empty response on attempt {attempt + 1}, retrying...")
                        continue

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error on attempt {attempt + 1}: {str(e)}, retrying...")
                        continue
                    raise

            return ""  # Return empty string if all retries fail

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return ""

    def bind_tools(self, tools: List[Any]) -> None:
        """Bind tools to the LLM with proper configuration"""
        # Convert tools to the format expected by Ollama
        self.llm = self.llm.bind_tools(
            tools,
            tool_choice="auto",  # Let the model decide which tool to use
        )


# Create a global instance
llm_manager = LLMManager()
