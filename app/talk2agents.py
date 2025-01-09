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

        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []

        if "messages" not in st.session_state:
            st.session_state.messages = []

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
                margin-bottom: 100px; /* Space for fixed input */
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
            }
            .debug-message.api { background-color: #e3f2fd; }
            .debug-message.error { background-color: #ffebee; }
            .debug-message.info { background-color: #f0f2f6; }
            </style>
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
            if st.button("Clear Debug Log"):
                st.session_state.debug_messages = []
            for msg in st.session_state.debug_messages:
                st.markdown(
                    f"""<div class="debug-message {msg['type']}">{msg['message']}</div>""",
                    unsafe_allow_html=True,
                )

    def render_papers(self, state: AgentState):
        """Render paper results"""
        if state.search_context.results and state.search_context.total_results > 0:
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

    def run(self):
        """Run the dashboard application"""
        self.setup_page()
        self.render_debug_panel()

        # Main chat container with bottom padding for fixed input
        chat_container = st.container()

        with chat_container:
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    # If this is a system message with search results, show papers
                    if message["role"] == "system" and "Found" in message.get(
                        "content", ""
                    ):
                        if hasattr(message, "papers"):
                            self.render_papers(message["papers"])

            st.markdown("</div>", unsafe_allow_html=True)

        # Fixed chat input at bottom
        with st.container():
            if prompt := st.chat_input("Message Academic Research Assistant..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Debug log
                self.add_debug_message(f"User Input: {prompt}", "info")

                # Process command
                with st.spinner("Thinking..."):
                    try:
                        self.add_debug_message("Processing command...", "info")
                        state = asyncio.run(self.manager.process_command_async(prompt))

                        # Debug log after processing
                        self.add_debug_message(
                            f"Command processed with status: {state.status}", "info"
                        )

                        # Get and show response
                        response_message = state.memory.messages[-1]
                        st.session_state.messages.append(response_message)

                        # If search was successful, attach papers to message
                        if (
                            state.status == AgentStatus.SUCCESS
                            and state.search_context.results
                        ):
                            self.add_debug_message(
                                f"Found {len(state.search_context.results)} papers",
                                "api",
                            )
                            response_message["papers"] = state

                    except Exception as e:
                        error_msg = {
                            "role": "system",
                            "content": f"I apologize, but I encountered an error: {str(e)}",
                        }
                        st.session_state.messages.append(error_msg)
                        self.add_debug_message(f"Error: {str(e)}", "error")

                # Rerun to update the UI
                st.rerun()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
