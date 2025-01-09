import asyncio
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus


class DashboardApp:
    def __init__(self):
        self.manager = EnhancedWorkflowManager()

    def setup_page(self):
        """Setup page configuration and styling"""
        st.set_page_config(
            page_title="Academic Paper Search", page_icon="📚", layout="wide"
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .paper-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid #e9ecef;
        }
        .chat-container {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 10px;
            background-color: #ffffff;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Header
        st.markdown(
            """
            <h1 style='text-align: center; color: #1e88e5; margin-bottom: 1rem;'>
                📚 Academic Research Assistant
            </h1>
            <p style='text-align: center; color: #666; margin-bottom: 2rem;'>
                Chat with me about research or ask me to find academic papers!
            </p>
        """,
            unsafe_allow_html=True,
        )

    def render_papers(self, state: AgentState):
        """Render paper results"""
        if state.search_context.results and state.search_context.total_results > 0:
            # Results header
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; 
                background-color: #e3f2fd; border-radius: 10px; margin-bottom: 2rem;'>
                    <h3 style='color: #1e88e5; margin: 0;'>
                        Found {state.search_context.total_results} papers
                    </h3>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Display papers
            for i, paper in enumerate(state.search_context.results, 1):
                with st.container():
                    st.markdown(
                        f"""
                        <div class='paper-card'>
                            <h3>{i}. {paper.title}</h3>
                            <p>👥 Authors: {', '.join(author['name'] for author in paper.authors)}</p>
                            <p>📅 Year: {paper.year} | 📊 Citations: {paper.citations}</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if paper.abstract:
                        with st.expander("Show Abstract"):
                            st.write(paper.abstract)

    def initialize_session_state(self):
        """Initialize session state for conversation history"""
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_chat_message(self, message: dict):
        """Render a chat message"""
        if message.get("content"):
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def run(self):
        """Run the dashboard application"""
        self.setup_page()
        self.initialize_session_state()

        # Chat interface
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        # Display chat history from session state
        for message in st.session_state.messages:
            self.render_chat_message(message)

        # Chat input
        if prompt := st.chat_input("Chat with me or ask me to search for papers..."):
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.render_chat_message({"role": "user", "content": prompt})

            # Process command
            with st.spinner("Thinking..."):
                try:
                    state = asyncio.run(self.manager.process_command_async(prompt))

                    # Get the response message
                    response_message = state.memory.messages[-1]

                    # Add to session state and display
                    st.session_state.messages.append(response_message)
                    self.render_chat_message(response_message)

                    # If this was a successful search, display papers
                    if (
                        state.status == AgentStatus.SUCCESS
                        and state.search_context.results
                    ):
                        self.render_papers(state)

                except Exception as e:
                    error_msg = {
                        "role": "system",
                        "content": f"I apologize, but I encountered an error: {str(e)}",
                    }
                    st.session_state.messages.append(error_msg)
                    self.render_chat_message(error_msg)

        st.markdown("</div>", unsafe_allow_html=True)


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
