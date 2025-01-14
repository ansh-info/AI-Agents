import asyncio
import json
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
        if "health_status" not in st.session_state:
            st.session_state.health_status = None

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

    def add_debug_message(self, message: str):
        """Add a debug message that will be displayed in the UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.debug_messages.append(f"[{timestamp}] {message}")

    def render_debug_panel(self):
        """Render debug panel in the sidebar with health status"""
        with st.sidebar:
            st.markdown("### 🔍 Debug Panel")

            # Add health check button
            if st.button("Check System Health"):
                with st.spinner("Checking system health..."):
                    st.session_state.health_status = asyncio.run(
                        st.session_state.workflow_manager.check_workflow_health()
                    )

            # Display health status if available
            if st.session_state.health_status:
                st.markdown("#### System Health")
                health = st.session_state.health_status

                # Display status with colored indicators
                st.markdown("**Components Status:**")
                st.markdown(
                    f"- Workflow Graph: {'✅' if health['workflow_graph'] else '❌'}"
                )
                st.markdown(
                    f"- Ollama Client: {'✅' if health['ollama_client'] else '❌'}"
                )
                st.markdown(
                    f"- Semantic Scholar: {'✅' if health['semantic_scholar'] else '❌'}"
                )
                st.markdown(
                    f"- Command Parser: {'✅' if health['command_parser'] else '❌'}"
                )

                # Display any errors
                if health["errors"]:
                    st.markdown("**Errors:**")
                    for error in health["errors"]:
                        st.error(error)

            # Debug messages section
            st.markdown("#### Debug Messages")
            if st.button("Clear Debug Log"):
                st.session_state.debug_messages = []

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

        for i, paper in enumerate(papers, 1):
            with st.container():
                # Title with numbering
                st.markdown(f"### {i}. {paper.title}")

                # Authors in a separate section
                st.markdown("**Authors:**")
                authors = ", ".join(author.get("name", "") for author in paper.authors)
                st.write(authors)

                # Publication details in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Year:**")
                    st.write(paper.year or "N/A")
                with col2:
                    st.markdown("**Citations:**")
                    st.write(paper.citations or 0)

                # Abstract in an expander
                if paper.abstract:
                    with st.expander("View Abstract"):
                        st.write(paper.abstract)

                # URL as a button
                if paper.url:
                    st.markdown(f"[View Paper]({paper.url})")

                # Add view details button
                if st.button("View Details", key=f"{context_prefix}_select_{i}"):
                    st.session_state.selected_paper = paper.paper_id
                    st.experimental_rerun()

                # Separator between papers
                st.divider()

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
        """Process user input with enhanced monitoring"""
        try:
            print(f"[DEBUG] Processing input: {prompt}")

            # Process command
            state = await st.session_state.workflow_manager.process_command_async(
                prompt
            )

            # Get debug info about the state
            debug_info = (
                await st.session_state.workflow_manager.workflow_graph.debug_state(
                    state
                )
            )
            print("[DEBUG] Current state info:")
            print(json.dumps(debug_info, indent=2))

            # Update session state
            st.session_state.agent_state = state

            if state.error_message:
                print(f"[DEBUG] Error in state: {state.error_message}")
                self.add_debug_message(f"Error: {state.error_message}")

            # Get and show response
            if state.memory and state.memory.messages:
                response_message = {
                    "role": "system",
                    "content": state.memory.messages[-1]["content"],
                }
                st.session_state.messages.append(response_message)

            return state

        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            self.add_debug_message(error_msg)
            st.session_state.messages.append(
                {
                    "role": "system",
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                }
            )
            return None

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
        """Render chat message history with enhanced paper formatting"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if (
                    message["role"] == "system"
                    and "Found" in message["content"]
                    and "papers related to" in message["content"]
                ):
                    # Split content into sections
                    sections = message["content"].split("\n\n")

                    # Display header (Found X papers...)
                    st.write(sections[0])

                    # Process each paper section
                    paper_sections = []
                    current_section = []

                    for section in sections[1:]:
                        if section.strip().startswith(("Summary:", "Based on")):
                            if current_section:
                                paper_sections.append("\n".join(current_section))
                            st.markdown("### Summary")
                            st.write(section.replace("Summary:", "").strip())
                            break
                        elif section.strip():
                            current_section.append(section)
                        else:
                            if current_section:
                                paper_sections.append("\n".join(current_section))
                                current_section = []

                    # Render each paper in a structured format
                    for i, paper in enumerate(paper_sections, 1):
                        with st.container():
                            lines = paper.split("\n")

                            # Extract title
                            title = (
                                lines[0].split("Authors:")[0].strip() if lines else ""
                            )
                            st.markdown(f"### {i}. {title}")

                            # Create two columns for metadata
                            col1, col2 = st.columns(2)

                            with col1:
                                # Extract and display authors
                                authors = next(
                                    (
                                        l.split("Authors:")[1].strip()
                                        for l in lines
                                        if "Authors:" in l
                                    ),
                                    "",
                                )
                                st.markdown("**Authors**")
                                st.write(authors)

                            with col2:
                                # Extract and display year/citations
                                year_citations = next(
                                    (
                                        l
                                        for l in lines
                                        if "Year:" in l and "Citations:" in l
                                    ),
                                    "",
                                )
                                if year_citations:
                                    st.markdown("**Publication Details**")
                                    st.write(year_citations)

                            # Display abstract in expander
                            abstract = next(
                                (
                                    l.split("Abstract:")[1].strip()
                                    for l in lines
                                    if "Abstract:" in l
                                ),
                                "",
                            )
                            if abstract:
                                with st.expander("View Abstract"):
                                    st.write(abstract)

                            # Display URL as link
                            url = next(
                                (
                                    l.split("URL:")[1].strip()
                                    for l in lines
                                    if "URL:" in l
                                ),
                                "",
                            )
                            if url:
                                st.markdown(f"[View Paper]({url})")

                            st.divider()
                else:
                    # Display regular messages
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
        """Run the dashboard application with initialization"""
        self.setup_page()

        # Initialize system asynchronously
        if "initialized" not in st.session_state:
            with st.spinner("Initializing system..."):
                asyncio.run(self.initialize_system())
                st.session_state.initialized = True

        # Render debug panel with health status
        self.render_debug_panel()

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
