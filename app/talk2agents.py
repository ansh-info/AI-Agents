import asyncio
import os
import sys
from datetime import datetime
from typing import List, Optional

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus, PaperContext

# Set page config at the very start of the script
st.set_page_config(
    page_title="Academic Paper Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "selected_paper" not in st.session_state:
            st.session_state.selected_paper = None

    def setup_page(self):
        """Setup page styling"""
        st.markdown(
            """
            <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .chat-message {
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
            }
            .chat-message.user {
                background-color: #e9ecef;
            }
            .paper-list {
                list-style-type: decimal;
                padding-left: 1.5rem;
            }
            .paper-item {
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: white;
                border: 1px solid #dee2e6;
            }
            .paper-title {
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .stChatInputContainer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: white;
                padding: 1rem 2rem;
                border-top: 1px solid #dee2e6;
                z-index: 100;
            }
            .main-content {
                margin-bottom: 80px;  /* Space for fixed chat input */
                padding: 20px;
            }
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

            # Add client status indicators
            if st.button("Check API Status"):
                with st.spinner("Checking API status..."):
                    health = asyncio.run(
                        st.session_state.workflow_manager.check_clients_health()
                    )
                    st.write("API Status:")
                    st.write(f"- Ollama: {'✅' if health['ollama_status'] else '❌'}")
                    st.write(
                        f"- Semantic Scholar: {'✅' if health['semantic_scholar_status'] else '❌'}"
                    )

            if st.button("Clear Debug Log"):
                st.session_state.debug_messages = []

            # Display debug messages
            for msg in st.session_state.debug_messages:
                st.markdown(
                    f"""<div class="debug-message {msg['type']}">
                    [{msg['timestamp']}] {msg['message']}
                    </div>""",
                    unsafe_allow_html=True,
                )

    def render_filters(self):
        """Render search filters"""
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                year = st.text_input("Year (e.g., 2023):")
                min_citations = st.number_input("Minimum Citations:", min_value=0)
            with col2:
                sort_by = st.selectbox(
                    "Sort by:", ["Relevance", "Citations", "Year"], key="sort_by"
                )
                is_open_access = st.checkbox("Open Access Only")

            return {
                "year": year if year else None,
                "min_citations": min_citations if min_citations > 0 else None,
                "sort_by": sort_by,
                "is_open_access": is_open_access,
            }

    def render_papers(self, papers: List[PaperContext], context_prefix: str = "main"):
        """Render paper results with enhanced display"""
        if not papers:
            st.info("No papers found. Try adjusting your search terms.")
            return

        st.markdown(f"<div class='paper-list'>", unsafe_allow_html=True)

        for i, paper in enumerate(papers, 1):
            st.markdown(
                f"""<div class='paper-item'>
                        <div class='paper-title'>{i}. {paper.title}</div>
                        <div><strong>Authors:</strong> {', '.join(author.get('name', '') for author in paper.authors)}</div>
                        <div><strong>Year:</strong> {paper.year or 'N/A'} | <strong>Citations:</strong> {paper.citations or 0}</div>
                        <div><strong>Abstract:</strong> {paper.abstract if paper.abstract else 'Not available'}</div>
                        </div>""",
                unsafe_allow_html=True,
            )

            if st.button("View Details", key=f"{context_prefix}_select_{i}"):
                st.session_state.selected_paper = paper.paper_id
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    def render_paper_details(self, paper: PaperContext):
        """Render detailed paper view"""
        st.markdown("## Paper Details")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {paper.title}")
            st.write(
                f"**Authors:** {', '.join(author.get('name', '') for author in paper.authors)}"
            )
            st.write(f"**Year:** {paper.year or 'N/A'}")
            st.write(f"**Citations:** {paper.citations or 0}")

            if paper.abstract:
                st.markdown("### Abstract")
                st.write(paper.abstract)

        with col2:
            if st.button("← Back to Results"):
                st.session_state.selected_paper = None
                st.experimental_rerun()

    async def process_input(self, prompt: str):
        """Process user input asynchronously"""
        try:
            # Process command
            state = await st.session_state.workflow_manager.process_command_async(
                prompt
            )

            # Update session state
            st.session_state.agent_state = state

            # Get and show response
            if state.memory and state.memory.messages:
                response_message = {
                    "role": "system",
                    "content": state.memory.messages[-1]["content"],
                }

                # If search was successful, attach papers to message
                if (
                    state.status == AgentStatus.SUCCESS
                    and state.search_context
                    and state.search_context.results
                ):
                    response_message["papers"] = state.search_context.results

                st.session_state.messages.append(response_message)

            return state

        except Exception as e:
            error_msg = {
                "role": "system",
                "content": f"I apologize, but I encountered an error: {str(e)}",
            }
            st.session_state.messages.append(error_msg)
            return None

    def render_chat_history(self):
        """Render chat message history"""
        st.markdown("<div class='main-content'>", unsafe_allow_html=True)

        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if (
                    message["role"] == "system"
                    and st.session_state.agent_state.search_context
                    and st.session_state.agent_state.search_context.results
                ):
                    self.render_papers(
                        st.session_state.agent_state.search_context.results,
                        context_prefix=f"chat_{idx}",
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    def render_pagination(self, total_results: int):
        """Render pagination controls"""
        if total_results > 0:
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                if st.session_state.current_page > 1:
                    if st.button("← Previous"):
                        st.session_state.current_page -= 1
                        st.rerun()  # Changed from experimental_rerun
            with col2:
                st.write(f"Page {st.session_state.current_page}")
            with col3:
                if len(st.session_state.agent_state.search_context.results) >= 10:
                    if st.button("Next →"):
                        st.session_state.current_page += 1
                        st.rerun()  # Changed from experimental_rerun

    def run(self):
        """Run the dashboard application"""
        self.setup_page()

        # Main content container
        st.markdown("<div class='main-content'>", unsafe_allow_html=True)

        # Only render chat history once
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # Only render papers for system messages with search results
                if (
                    message["role"] == "system"
                    and hasattr(message, "papers")
                    and message.get("papers")
                ):
                    self.render_papers(message["papers"], context_prefix=f"chat_{idx}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Fixed chat input at bottom
        with st.container():
            if prompt := st.chat_input("Ask about research papers..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Process command
                with st.spinner("Processing..."):
                    response_state = asyncio.run(self.process_input(prompt))

                st.rerun()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
