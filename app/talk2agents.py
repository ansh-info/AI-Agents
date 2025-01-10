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
        """Initialize the dashboard application"""
        if "workflow_manager" not in st.session_state:
            st.session_state.workflow_manager = EnhancedWorkflowManager()
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = AgentState()
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def setup_page(self):
        """Setup page configuration and styling"""
        st.set_page_config(
            page_title="Academic Paper Search",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS for chat interface
        st.markdown(
            """
            <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .element-container {
                width: 100%;
            }
            .stChatInputContainer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 1rem 2rem;
                background: white;
                border-top: 1px solid #ddd;
                z-index: 100;
            }
            .chat-messages {
                margin-bottom: 100px;
                padding: 20px;
            }
            .paper-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border: 1px solid #e9ecef;
            }
            .debug-message { 
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-family: monospace;
            }
            .debug-message.api { background-color: #f0f8ff; }
            .debug-message.error { background-color: #fff0f0; }
            .debug-message.info { background-color: #f0fff0; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def add_debug_message(self, message: str, msg_type: str = "info"):
        """Add a debug message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_msg = {"timestamp": timestamp, "message": message, "type": msg_type}
        st.session_state.debug_messages.append(debug_msg)

    def render_debug_panel(self):
        """Render debug panel in sidebar"""
        with st.sidebar:
            st.markdown("### 🔍 Debug Panel")
            if st.button("Clear Debug Log"):
                st.session_state.debug_messages = []

            for msg in st.session_state.debug_messages:
                st.markdown(
                    f"""<div class="debug-message {msg['type']}">
                    [{msg['timestamp']}] {msg['message']}
                    </div>""",
                    unsafe_allow_html=True,
                )

    def render_papers(self, papers):
        """Render paper results"""
        for i, paper in enumerate(papers, 1):
            with st.container():
                st.markdown(
                    f"""
                    <div class="paper-card">
                        <h3>{i}. {paper.title}</h3>
                        <p><strong>Year:</strong> {paper.year or 'N/A'}</p>
                        <p><strong>Citations:</strong> {paper.citations or 0}</p>
                        <p><strong>Authors:</strong> {', '.join(author.get('name', '') for author in paper.authors)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if paper.abstract:
                    with st.expander("Show Abstract"):
                        st.write(paper.abstract)

    def run(self):
        """Run the dashboard application"""
        self.setup_page()
        self.render_debug_panel()

        # Main chat container
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "system" and hasattr(message, "papers"):
                    self.render_papers(message["papers"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Message Academic Research Assistant..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Debug logging
            self.add_debug_message(f"User Input: {prompt}", "info")

            # Process command
            with st.spinner("Processing..."):
                try:
                    # Debug log before processing
                    self.add_debug_message("Starting command processing...", "info")

                    # Process command
                    state = asyncio.run(
                        st.session_state.workflow_manager.process_command_async(prompt)
                    )

                    # Update session state
                    st.session_state.agent_state = state

                    # Debug log after processing
                    self.add_debug_message(
                        f"Command processed with status: {state.status}. Current step: {state.current_step}",
                        "info",
                    )

                    # Add state details to debug
                    if state.search_context and state.search_context.results:
                        self.add_debug_message(
                            f"Search results: Found {len(state.search_context.results)} papers",
                            "api",
                        )

                    # Get and show response
                    if state.memory and state.memory.messages:
                        response_message = state.memory.messages[-1]
                        st.session_state.messages.append(response_message)

                        # If search was successful, attach papers to message
                        if (
                            state.status == AgentStatus.SUCCESS
                            and state.search_context.results
                        ):
                            response_message["papers"] = state.search_context.results

                except Exception as e:
                    error_msg = {
                        "role": "system",
                        "content": f"I apologize, but I encountered an error: {str(e)}",
                    }
                    st.session_state.messages.append(error_msg)
                    self.add_debug_message(f"Error: {str(e)}", "error")

            # Rerun to update UI
            st.rerun()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    from datetime import datetime

    main()
