# In utils/llm.py

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config.config import config
from state.shared_state import shared_state


def create_llm() -> ChatOllama:
    """Create and configure Ollama LLM instance"""
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=0,  # Set to 0 for most deterministic output
        stop=["\n\n", "</function>", "```"],  # Better stop tokens
        frequency_penalty=0,  # Reduce repetition
        presence_penalty=0,  # Reduce tangential content
        top_p=0.1,  # More focused sampling
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

            # Get response with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(messages)
                    if response and response.content.strip():
                        return response.content
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
        """Bind tools to the LLM"""
        self.llm = self.llm.bind_tools(tools)


# Create a global instance
llm_manager = LLMManager()
