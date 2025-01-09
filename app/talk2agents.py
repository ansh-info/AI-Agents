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
            page_title="Academic Paper Search",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize debug state
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []

        # Custom CSS
        st.markdown(
            """
        <style>
        .debug-message { 
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: #f0f2f6;
        }
        .debug-message.api { background-color: #e3f2fd; }
        .debug-message.error { background-color: #ffebee; }
        .debug-message.info { background-color: #f0f2f6; }
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
            """,
            unsafe_allow_html=True,
        )

    def add_debug_message(self, message: str, msg_type: str = "info"):
        """Add a debug message to the session state"""
        st.session_state.debug_messages.append({"message": message, "type": msg_type})

    def render_debug_panel(self):
        """Render debug panel in sidebar"""
        with st.sidebar:
            st.markdown("### 🔍 Debug Panel")

            # Add clear button
            if st.button("Clear Debug Log"):
                st.session_state.debug_messages = []

            # Show debug messages
            for msg in st.session_state.debug_messages:
                st.markdown(
                    f"""<div class="debug-message {msg['type']}">
                        {msg['message']}
                    </div>""",
                    unsafe_allow_html=True,
                )

    def render_papers(self, state: AgentState):
        """Render paper results"""
        if state.search_context.results and state.search_context.total_results > 0:
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

    def render_chat_message(self, message: dict):
        """Render a chat message"""
        if message.get("content"):
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def run(self):
        """Run the dashboard application"""
        self.setup_page()
        self.render_debug_panel()

        # Main content area
        with st.container():
            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                self.render_chat_message(message)

            # Chat input
            if prompt := st.chat_input(
                "Chat with me or ask me to search for papers..."
            ):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                self.render_chat_message({"role": "user", "content": prompt})

                # Debug log
                self.add_debug_message(f"User Input: {prompt}", "info")

                # Process command
                with st.spinner("Thinking..."):
                    try:
                        # Debug log before processing
                        self.add_debug_message("Processing command...", "info")

                        state = asyncio.run(self.manager.process_command_async(prompt))

                        # Debug log after processing
                        self.add_debug_message(
                            f"Command processed with status: {state.status}", "info"
                        )

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
                            self.add_debug_message(
                                f"Found {len(state.search_context.results)} papers",
                                "api",
                            )
                            self.render_papers(state)

                    except Exception as e:
                        error_msg = {
                            "role": "system",
                            "content": f"I apologize, but I encountered an error: {str(e)}",
                        }
                        st.session_state.messages.append(error_msg)
                        self.render_chat_message(error_msg)
                        self.add_debug_message(f"Error: {str(e)}", "error")


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
