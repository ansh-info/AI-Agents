import asyncio
import os
import re
import sys
from datetime import datetime
from typing import List

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent.enhanced_workflow import EnhancedWorkflowManager
from state.agent_state import AgentState, PaperContext


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
        """Add a debug message to the UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.debug_messages.append(f"[{timestamp}] {message}")

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

            # Store the current paper being discussed if it's a paper request
            if "paper" in prompt.lower():
                try:
                    numbers = re.findall(r"\d+", prompt)
                    if numbers:
                        st.session_state.current_paper_number = int(numbers[0]) - 1
                except ValueError:
                    pass

            state = await st.session_state.workflow_manager.process_command_async(
                prompt
            )
            st.session_state.agent_state = state

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Add system response if available
            if state and state.memory and state.memory.messages:
                last_msg = state.memory.messages[-1]
                if last_msg["role"] == "system":
                    # Handle search queries
                    if (
                        state.search_context
                        and state.search_context.results
                        and any(
                            keyword in prompt.lower()
                            for keyword in [
                                "find",
                                "search",
                                "papers on",
                                "papers about",
                            ]
                        )
                    ):
                        formatted_content = self.format_paper_results(
                            state.search_context.results
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": formatted_content,
                                "type": "search_results",
                            }
                        )
                        st.session_state.last_search_results = (
                            state.search_context.results
                        )

                    # Handle paper-specific queries
                    elif hasattr(st.session_state, "current_paper_number") and (
                        "paper" in prompt.lower()
                        or any(
                            keyword in prompt.lower()
                            for keyword in [
                                "summarize",
                                "summarise",
                                "tell me more",
                                "explain",
                            ]
                        )
                    ):
                        paper_number = st.session_state.current_paper_number
                        if hasattr(
                            st.session_state, "last_search_results"
                        ) and 0 <= paper_number < len(
                            st.session_state.last_search_results
                        ):
                            paper = st.session_state.last_search_results[paper_number]

                            if (
                                "summarise" in prompt.lower()
                                or "summarize" in prompt.lower()
                            ):
                                # Create analysis prompt for the LLM
                                analysis_prompt = "Based on this academic paper:\n\n"
                                analysis_prompt += f"Title: {paper.title}\n"
                                analysis_prompt += f"Abstract: {paper.abstract if paper.abstract else 'No abstract available'}\n\n"
                                analysis_prompt += (
                                    "Please provide a detailed analysis covering:\n"
                                )
                                analysis_prompt += "1. Main theme and key objectives\n"
                                analysis_prompt += "2. Key research contributions\n"
                                analysis_prompt += "3. Methodology or approach\n"
                                analysis_prompt += "4. Significant findings\n"
                                analysis_prompt += (
                                    "5. Potential impact and applications\n"
                                )
                                analysis_prompt += "\nPresent the analysis in a clear, bulleted format."

                                # Get analysis from LLM
                                analysis_state = await st.session_state.workflow_manager.process_command_async(
                                    analysis_prompt
                                )
                                analysis = "Unable to generate detailed analysis."
                                if (
                                    analysis_state
                                    and analysis_state.memory
                                    and analysis_state.memory.messages
                                ):
                                    analysis = analysis_state.memory.messages[-1][
                                        "content"
                                    ]

                                # Format enhanced summary response
                                response = (
                                    "### Summary of Paper "
                                    + str(paper_number + 1)
                                    + "\n\n"
                                )
                                response += f"**Title:** {paper.title}\n"
                                response += f"**Authors:** {', '.join(author.get('name', '') for author in paper.authors)}\n\n"
                                response += "**Publication Details:**\n"
                                response += f"- Year: {paper.year or 'N/A'}\n"
                                response += f"- Citations: {paper.citations or 0}\n\n"
                                response += "**Analysis:**\n"
                                response += f"{analysis}\n\n"
                                response += f"**Source:** {paper.url if paper.url else 'URL not available'}\n\n"
                                response += "Would you like to explore any specific aspect of this paper in more detail?"

                            else:
                                # Format detailed response
                                response = (
                                    "Here are the details for paper "
                                    + str(paper_number + 1)
                                    + ":\n\n"
                                )
                                response += f"Title: {paper.title}\n"
                                response += f"Authors: {', '.join(author.get('name', '') for author in paper.authors)}\n"
                                response += f"Year: {paper.year or 'N/A'}\n"
                                response += f"Citations: {paper.citations or 0}\n\n"
                                response += f"Abstract: {paper.abstract if paper.abstract else 'No abstract available'}\n\n"
                                response += f"URL: {paper.url if paper.url else 'No URL available'}"

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": response,
                                    "type": "paper_details",
                                }
                            )
                        else:
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "I'm sorry, but I couldn't find the paper you're referring to. Could you please specify which paper you'd like to know about?",
                                    "type": "error",
                                }
                            )
                    else:
                        # Regular response
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": last_msg["content"],
                                "type": "regular",
                            }
                        )

            return state

        except Exception as e:
            print(f"[DEBUG] Error in dashboard: {str(e)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "type": "error",
                }
            )
            return None

    def format_paper_results(self, papers: List[PaperContext]) -> str:
        """Format paper results with both styled list and table view"""
        # First part: Styled list view (as before)
        css = """
        <style>
        .paper-title {
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 1em;
            color: #1a1a1a;
        }
        .paper-metadata {
            font-size: 1em;
            margin-bottom: 0.8em;
        }
        .paper-metadata-label {
            font-weight: bold;
        }
        .paper-abstract {
            font-size: 1.1em;
            margin: 1em 0;
        }
        .paper-abstract-label {
            font-weight: bold;
            display: block;
            margin-bottom: 0.5em;
        }
        .paper-link {
            font-size: 0.9em;
            color: #4169E1;
            text-decoration: none;
            margin-top: 1em;
            display: inline-block;
        }
        .paper-divider {
            margin: 2em 0;
            border-bottom: 1px solid #eee;
        }
        .table-view {
            margin-top: 2em;
        }
        </style>
        """

        # List view
        formatted_results = [css]

        for i, paper in enumerate(papers, 1):
            paper_text = [
                f'<div class="paper-title">{i}. {paper.title}</div>',
                '<div class="paper-metadata">',
                f'<span class="paper-metadata-label">Authors:</span> {", ".join(author.get("name", "") for author in paper.authors)}<br>',
                f'<span class="paper-metadata-label">Year:</span> {paper.year or "N/A"} | <span class="paper-metadata-label">Citations:</span> {paper.citations or 0}',
                "</div>",
                '<div class="paper-abstract">',
                '<span class="paper-abstract-label">Abstract:</span>',
                f"{paper.abstract if paper.abstract else 'No abstract available'}",
                "</div>",
                (
                    f'<a href="{paper.url}" class="paper-link">View Paper</a>'
                    if paper.url
                    else ""
                ),
                '<div class="paper-divider"></div>',
            ]
            formatted_results.append("\n".join(paper_text))

        # Create DataFrame for table view
        paper_data = [
            {
                "Title": paper.title,
                "Authors": ", ".join(
                    author.get("name", "") for author in paper.authors
                ),
                "Year": paper.year or "N/A",
                "Citations": paper.citations or 0,
                "URL": f'<a href="{paper.url}">View Paper</a>' if paper.url else "",
            }
            for paper in papers
        ]

        # Convert to DataFrame
        import pandas as pd

        df = pd.DataFrame(paper_data)

        # Add DataFrame to session state for access in render method
        if "paper_table" not in st.session_state:
            st.session_state.paper_table = df
        else:
            st.session_state.paper_table = df

        return "\n".join(formatted_results)

    def render_chat_interface(self):
        """Render chat interface with both list and table views"""
        st.title("Talk2Comp - Litrature Research Assistant")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    msg_type = message.get("type", "regular")

                    if msg_type == "search_results":
                        # Render the formatted list view
                        st.markdown(message["content"], unsafe_allow_html=True)

                        # Add table view if available
                        if (
                            hasattr(st.session_state, "paper_table")
                            and not st.session_state.paper_table.empty
                        ):
                            st.markdown("### Table View of Results")
                            st.dataframe(
                                st.session_state.paper_table,
                                column_config={
                                    "Title": st.column_config.TextColumn(
                                        "Title", width="large"
                                    ),
                                    "Authors": st.column_config.TextColumn(
                                        "Authors", width="medium"
                                    ),
                                    "Year": st.column_config.NumberColumn(
                                        "Year", width="small"
                                    ),
                                    "Citations": st.column_config.NumberColumn(
                                        "Citations", width="small"
                                    ),
                                    "URL": st.column_config.LinkColumn(
                                        "Link", width="small"
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )
                    else:
                        # Regular response - just show the message content
                        st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    st.write(message["content"])

        if prompt := st.chat_input("Ask me anything about research papers..."):
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Processing..."):
                asyncio.run(self.process_input(prompt))

            st.rerun()

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
        # Render chat interface
        self.render_chat_interface()


def main():
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
