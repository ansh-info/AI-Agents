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
        # Custom CSS for enhanced UI
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
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .paper-card:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            .debug-message { 
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-family: monospace;
            }
            .debug-message.api { background-color: #403f3e; color: #fff; }
            .debug-message.error { background-color: #403f3e; color: #ff6b6b; }
            .debug-message.info { background-color: #403f3e; color: #69db7c; }
            .filter-section {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            .pagination {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 1rem 0;
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

    def render_papers(self, papers: List[PaperContext]):
        """Render paper results with enhanced display"""
        if not papers:
            st.info("No papers found. Try adjusting your search terms.")
            return

        st.write(f"Found {len(papers)} papers:")

        for i, paper in enumerate(papers, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"""<div class="paper-card">
                                <h3>{i}. {paper.title}</h3>
                                <p><strong>Authors:</strong> {', '.join(author.get('name', '') for author in paper.authors)}</p>
                                <p><strong>Year:</strong> {paper.year or 'N/A'} | <strong>Citations:</strong> {paper.citations or 0}</p>
                                {f'<p><strong>Abstract:</strong> {paper.abstract}</p>' if paper.abstract else ''}
                            </div>""",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("View Details", key=f"select_{i}"):
                        st.session_state.selected_paper = paper.paper_id
                        st.rerun()  # Changed from experimental_rerun

                if paper.abstract:
                    with st.expander("Show Abstract"):
                        st.write(paper.abstract)

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
            # Debug logging
            self.add_debug_message(f"Processing input: {prompt}", "info")

            # Check client health
            health = await st.session_state.workflow_manager.check_clients_health()
            if not health["all_healthy"]:
                self.add_debug_message("Attempting to reload clients...", "info")
                await st.session_state.workflow_manager.reload_clients()

            # Process command
            state = await st.session_state.workflow_manager.process_command_async(
                prompt
            )

            # Update session state
            st.session_state.agent_state = state

            # Debug log after processing
            self.add_debug_message(
                f"Command processed with status: {state.status}. Current step: {state.current_step}",
                "info",
            )

            # Log search results if available
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
                if state.status == AgentStatus.SUCCESS and state.search_context.results:
                    response_message["papers"] = state.search_context.results

            return state

        except Exception as e:
            error_msg = {
                "role": "system",
                "content": f"I apologize, but I encountered an error: {str(e)}",
            }
            st.session_state.messages.append(error_msg)
            self.add_debug_message(f"Error: {str(e)}", "error")
            return None

    def render_chat_history(self):
        """Render chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "system" and "papers" in message:
                    self.render_papers(message["papers"])

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
        self.render_debug_panel()

        # Main chat container
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

        # Render filters if we have search results
        if (
            st.session_state.agent_state.search_context
            and st.session_state.agent_state.search_context.results
        ):
            filters = self.render_filters()
            st.session_state.agent_state.search_context.current_filters = filters

        # Display chat history
        self.render_chat_history()

        # Render pagination if we have search results
        if (
            st.session_state.agent_state.search_context
            and st.session_state.agent_state.search_context.total_results > 0
        ):
            self.render_pagination(
                st.session_state.agent_state.search_context.total_results
            )

        # Selected paper view
        if st.session_state.selected_paper:
            paper = next(
                (
                    p
                    for p in st.session_state.agent_state.search_context.results
                    if p.paperId == st.session_state.selected_paper
                ),
                None,
            )
            if paper:
                self.render_paper_details(paper)

        st.markdown("</div>", unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Message Academic Research Assistant..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Process command
            with st.spinner("Processing..."):
                asyncio.run(self.process_input(prompt))

            # Rerun to update UI
            st.rerun()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
