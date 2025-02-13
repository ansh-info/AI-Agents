from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config.config import config
from state.shared_state import shared_state


def create_llm() -> ChatOllama:
    """Create and configure Ollama LLM instance"""
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=0.1,  # Lower temperature for more deterministic outputs
        stop=["</function>", "```"],  # Stop tokens to prevent extra content
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
        """
        Get response from LLM with system prompt and user input

        Args:
            system_prompt: The system prompt to guide the LLM
            user_input: The user's input/query
            additional_context: Optional additional context to append to user input
            include_history: Whether to include chat history in context

        Returns:
            str: The LLM's response
        """
        # Prepare context string if provided
        context_parts = []

        if additional_context:
            context_parts.append(
                "Context:\n"
                + "\n".join(f"{k}: {v}" for k, v in additional_context.items())
            )

        if include_history:
            recent_history = shared_state.get_chat_history(limit=5)
            if recent_history:
                history_str = "\nRecent Conversation:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in recent_history
                )
                context_parts.append(history_str)

        context_str = "\n\n".join(context_parts) if context_parts else ""

        # Combine user input with context
        full_input = user_input + context_str

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", full_input)]
        )

        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"

    def bind_tools(self, tools: List[Any]) -> None:
        """Bind tools to the LLM"""
        self.llm = self.llm.bind_tools(tools)


# Create a global instance
llm_manager = LLMManager()
