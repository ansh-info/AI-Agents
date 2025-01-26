import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.enhanced_workflow import EnhancedWorkflowManager
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
        """Initialize the dashboard application with new workflow"""
        if "workflow_manager" not in st.session_state:
            st.session_state.workflow_manager = EnhancedWorkflowManager()
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = AgentState()
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []

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
            }
            .chat-message.user {
                background-color: #e9ecef;
            }
            .chat-message.assistant {
                background-color: #f8f9fa;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def add_debug_message(self, message: str):
        """Add a debug message to the UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.debug_messages.append(f"[{timestamp}] {message}")

    def render_debug_panel(self):
        """Render debug panel"""
        with st.sidebar:
            st.markdown("### 🔍 Debug Panel")

            if st.button("Check System Health"):
                with st.spinner("Checking system health..."):
                    health_status = asyncio.run(
                        st.session_state.workflow_manager.check_workflow_health()
                    )
                    st.json(health_status)

            # Debug messages
            if st.session_state.debug_messages:
                st.markdown("### Debug Messages")
                for msg in st.session_state.debug_messages:
                    st.text(msg)

    async def initialize_system(self):
        """Initialize system and check health"""
        try:
            # Check system health on startup
            health_status = (
                await st.session_state.workflow_manager.check_workflow_health()
            )
            st.session_state.health_status = health_status

            # Log initialization status
            self.add_debug_message("System initialized")
            if not health_status["ollama_client"]:
                self.add_debug_message("WARNING: Ollama client not responding")
            if not health_status["semantic_scholar"]:
                self.add_debug_message("WARNING: Semantic Scholar API not responding")
        except Exception as e:
            self.add_debug_message(f"Error during initialization: {str(e)}")

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

        st.markdown("### Search Results")

        for i, paper in enumerate(papers, 1):
            with st.container():
                # Paper container with border and padding
                st.markdown(
                    """
                    <style>
                    .paper-container {
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                        background-color: #ffffff;
                    }
                    .paper-title {
                        font-size: 1.2rem;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .paper-metadata {
                        font-size: 0.9rem;
                        color: #666;
                        margin-bottom: 10px;
                    }
                    .paper-abstract {
                        font-size: 0.95rem;
                        margin: 10px 0;
                    }
                    .paper-url {
                        font-size: 0.9rem;
                    }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="paper-container">', unsafe_allow_html=True)

                # Title and number
                st.markdown(
                    f'<div class="paper-title">{i}. {paper.title}</div>',
                    unsafe_allow_html=True,
                )

                # Metadata
                authors = ", ".join(author.get("name", "") for author in paper.authors)
                st.markdown(
                    f'<div class="paper-metadata">**Authors:** {authors}<br>'
                    f"**Year:** {paper.year or 'N/A'} | **Citations:** {paper.citations or 0}</div>",
                    unsafe_allow_html=True,
                )

                # Abstract
                if paper.abstract:
                    with st.expander("Show Abstract"):
                        st.markdown(
                            f'<div class="paper-abstract">{paper.abstract}</div>',
                            unsafe_allow_html=True,
                        )

                # URL and actions
                if paper.url:
                    st.markdown(
                        f'<div class="paper-url">[View Paper]({paper.url})</div>',
                        unsafe_allow_html=True,
                    )

                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("View Details", key=f"{context_prefix}_select_{i}"):
                        st.session_state.selected_paper = paper.paper_id
                        st.experimental_rerun()

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
        """Process user input using workflow manager"""
        try:
            print(f"[DEBUG] Dashboard processing input: {prompt}")

            # Process through workflow
            state = await st.session_state.workflow_manager.process_command_async(
                prompt
            )

            # Update session state
            st.session_state.agent_state = state

            # Add messages to chat history
            if not isinstance(prompt, dict):
                st.session_state.messages.append({"role": "user", "content": prompt})

            if state and state.memory and state.memory.messages:
                # Get the last system message
                last_system_message = next(
                    (
                        msg
                        for msg in reversed(state.memory.messages)
                        if msg["role"] == "system"
                    ),
                    None,
                )

                if last_system_message:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": last_system_message["content"]}
                    )
                    print(
                        f"[DEBUG] Added response: {last_system_message['content'][:200]}..."
                    )

            return state

        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                }
            )
            return None

    def render_chat_interface(self):
        """Render chat interface"""
        st.title("Talk2Papers - Academic Research Assistant")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about research papers..."):
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Processing..."):
                asyncio.run(self.process_input(prompt))

    async def process_search(
        self, query: str, year: str = None, citations: str = None, sort_by: str = None
    ):
        """Process search query with state monitoring"""
        print(f"[DEBUG] Starting search for: {query}")
        try:
            cleaned_query = self.clean_search_query(query)
            search_command = f"search {cleaned_query}"

            # Add filters if provided
            if year:
                search_command += f" year:{year}"
            if citations:
                search_command += f" citations>{citations}"
            if sort_by and sort_by != "Relevance":
                search_command += f" sort:{sort_by.lower()}"

            print(f"[DEBUG] Executing command: {search_command}")

            state = await self.manager.process_command_async(search_command)

            # Get and log debug info
            debug_info = await self.manager.workflow_graph.debug_state(state)
            print("[DEBUG] Search completed. State info:")
            print(json.dumps(debug_info, indent=2))

            if state.status == AgentStatus.SUCCESS:
                st.session_state.current_results = state
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                st.session_state.current_page = 1

                # Store papers in context
                if state.search_context and state.search_context.results:
                    for paper in state.search_context.results:
                        st.session_state.paper_context[paper.paperId] = paper

            return state

        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
            self.add_debug_message(f"Search error: {str(e)}")
            return None

    def render_chat_history(self):
        """Render chat message history with enhanced formatting"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if (
                    message["role"] == "system"
                    and "# Search Results" in message["content"]
                ):
                    # Split content into sections
                    sections = message["content"].split("\n## ")

                    # Display header
                    st.markdown(sections[0])  # Main header

                    # Process remaining sections
                    for section in sections[1:]:
                        if section.startswith("Summary"):
                            # Display summary
                            st.markdown(f"## {section}")
                        elif section.strip():
                            # Display paper entries
                            paper_section = f"## {section}"
                            st.markdown(paper_section)
                else:
                    # Regular message display
                    st.write(message["content"])

    def _render_paper_section(self, paper_text):
        """Render a single paper section with proper formatting"""
        lines = paper_text.split("\n")
        if not lines:
            return

        # Extract title (first line)
        title = lines[0].split("Authors:")[0].strip()
        st.markdown(f"### {title}")

        # Create columns for metadata
        col1, col2 = st.columns(2)

        # Extract and display authors
        authors = next((line for line in lines if "Authors:" in line), "")
        if authors:
            authors = authors.split("Authors:")[1].strip()
            with col1:
                st.markdown("**Authors:**")
                st.write(authors)

        # Extract and display year and citations
        year_citations = next((line for line in lines if "Year:" in line), "")
        if year_citations:
            with col2:
                st.markdown("**Publication Details:**")
                st.write(year_citations)

        # Extract and display abstract in expander
        abstract = next((line for line in lines if "Abstract:" in line), "")
        if abstract:
            with st.expander("View Abstract"):
                st.write(abstract.split("Abstract:")[1].strip())

        # Extract and display URL
        url = next((line for line in lines if "URL:" in line), "")
        if url:
            st.markdown(
                f"**Link:** [{url.split('URL:')[1].strip()}]({url.split('URL:')[1].strip()})"
            )

        st.divider()

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
        self.render_chat_interface()

        if st.sidebar.checkbox("Show Debug Panel"):
            self.render_debug_panel()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
