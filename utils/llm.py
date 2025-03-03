from typing import Any, Dict, List, Union
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.config import config
from state.shared_state import shared_state


def create_llm() -> ChatOpenAI:
    """Create and configure OpenAI LLM instance"""
    return ChatOpenAI(
        model="gpt-4",  # You can change this to gpt-3.5-turbo for lower cost
        temperature=config.TEMPERATURE,
        timeout=60,  # Timeout in seconds
        max_retries=3,
        # Adjust these based on your needs
        model_kwargs={
            "top_p": 0.95,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )


class LLMManager:
    def __init__(self):
        # Initialize the LLM with default configuration
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
            # Create messages list
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]

            # Add chat history if requested
            if include_history:
                history = shared_state.get_chat_history(limit=3)
                for msg in history:
                    if msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))

            # Add debug logging
            print("\nDebug - LLM Input:")
            print(f"System prompt: {system_prompt[:200]}...")
            print(f"User input: {user_input}")

            # Get response with retries
            response = self.llm.invoke(messages)

            # Add debug logging
            print("\nDebug - LLM Response:")
            print(f"Raw response: {response.content}")

            if response and response.content.strip():
                return response.content.strip()

            return ""

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return ""

    def bind_tools(self, tools: List[Any]) -> None:
        """Bind tools to the LLM for function/tool calling"""
        self.llm = self.llm.bind_tools(
            tools, tool_choice="auto"  # Let the model decide which tool to use
        )


# Create a global instance
llm_manager = LLMManager()
