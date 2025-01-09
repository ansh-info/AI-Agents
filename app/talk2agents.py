import asyncio
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentStatus


class SearchAgent:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.s2_client = SemanticScholarClient()
        self.MAX_PAPERS = 10  # Fixed limit of 10 papers

    async def enhance_query(self, query: str) -> str:
        """Use LLM to enhance the search query"""
        prompt = f"""Enhance this academic search query: "{query}"
        Remove conversational phrases and add relevant academic terms.
        Keep it focused and return only the enhanced search terms.
        """
        try:
            enhanced = await self.ollama_client.generate(prompt=prompt, max_tokens=50)
            return enhanced.strip() or query
        except Exception as e:
            return query

    async def process_message(self, message: str, papers: list = None):
        """Process a conversational message"""
        if not papers:
            papers = []

        # Check if the message is referencing a specific paper
        if "paper" in message.lower() and any(
            str(i) for i in range(1, len(papers) + 1) if str(i) in message
        ):
            # Extract paper number
            for i in range(1, len(papers) + 1):
                if str(i) in message:
                    paper = papers[i - 1]
                    context = f"""
                    Title: {paper.title}
                    Authors: {', '.join(a.get('name', '') for a in paper.authors)}
                    Year: {paper.year}
                    Abstract: {paper.abstract or 'Not available'}
                    Citations: {paper.citationCount}
                    """
                    prompt = f"Given this paper information:\n{context}\n\nUser question: {message}\nProvide a helpful response:"
                    response = await self.ollama_client.generate(
                        prompt=prompt, max_tokens=300
                    )
                    return response.strip()

        # If it's a search query
        elif any(
            term in message.lower() for term in ["find", "search", "papers", "research"]
        ):
            results, error = await self.search_papers(message)
            if error:
                return f"Search error: {error}"
            if results and results.papers:
                response = "Here are the papers I found:\n\n"
                for i, paper in enumerate(results.papers[: self.MAX_PAPERS], 1):
                    response += f"{i}. {paper.title} ({paper.year}) - {paper.citationCount} citations\n"
                response += "\nYou can ask me about any specific paper by its number."
                return response, results.papers[: self.MAX_PAPERS]
            return "No papers found matching your criteria."

        # General conversation
        else:
            prompt = f"User message: {message}\nProvide a helpful response about academic papers:"
            response = await self.ollama_client.generate(prompt=prompt, max_tokens=200)
            return response.strip()

    async def search_papers(self, query: str) -> tuple:
        """Search for papers with rate limiting"""
        try:
            params = {"query": query, "limit": self.MAX_PAPERS, "offset": 0}

            # Add delay for rate limiting
            await asyncio.sleep(1)
            results = await self.s2_client.search_papers(**params)
            return results, None

        except Exception as e:
            return None, str(e)


class StreamlitApp:
    def __init__(self):
        self.agent = SearchAgent()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_papers" not in st.session_state:
            st.session_state.current_papers = None

    def render_chat_message(self, message):
        """Render a chat message"""
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.write(content)

    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(page_title="Academic Paper Search", layout="wide")
        self.initialize_session_state()

        # Custom CSS for better appearance
        st.markdown(
            """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stChatMessage {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Header
        st.markdown(
            """
            <h1 style='text-align: center; color: #1e88e5; margin-bottom: 1rem;'>
                📚 Academic Paper Assistant
            </h1>
        """,
            unsafe_allow_html=True,
        )

        # Chat interface
        for message in st.session_state.messages:
            self.render_chat_message(message)

        # Chat input
        if prompt := st.chat_input("Ask about papers or search for research..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.render_chat_message({"role": "user", "content": prompt})

            # Get assistant response
            with st.spinner("Thinking..."):
                response = asyncio.run(
                    self.agent.process_message(prompt, st.session_state.current_papers)
                )

                if isinstance(response, tuple):  # Search results
                    response_text, papers = response
                    st.session_state.current_papers = (
                        papers  # Store papers for reference
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    self.render_chat_message(
                        {"role": "assistant", "content": response_text}
                    )
                else:  # Normal response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    self.render_chat_message({"role": "assistant", "content": response})


def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
