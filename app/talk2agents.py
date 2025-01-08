import asyncio
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from state.agent_state import AgentStatus


class StreamlitApp:
    def __init__(self):
        self.manager = EnhancedWorkflowManager()

    def initialize_session_state(self):
        """Initialize session state variables if they don't exist"""
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        if "current_results" not in st.session_state:
            st.session_state.current_results = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

    def clean_search_query(self, query: str) -> str:
        """Clean and format the search query"""
        # Remove question words and common phrases
        remove_phrases = [
            "can you",
            "find me",
            "papers on",
            "papers about",
            "search for",
            "look for",
            "tell me about",
        ]
        cleaned_query = query.lower()
        for phrase in remove_phrases:
            cleaned_query = cleaned_query.replace(phrase, "")

        # Remove extra whitespace and trim
        cleaned_query = " ".join(cleaned_query.split())
        return cleaned_query

    async def process_search(
        self, query: str, year: str = None, citations: str = None, sort_by: str = None
    ):
        """Process search query and update state"""
        # Clean the query
        cleaned_query = self.clean_search_query(query)

        # Build the search command with filters
        search_command = f"search {cleaned_query}"
        if year:
            search_command += f" year:{year}"
        if citations:
            search_command += f" citations>{citations}"
        if sort_by and sort_by != "Relevance":
            search_command += f" sort:{sort_by.lower()}"

        # Log the search command for debugging
        st.write(f"Debug: Original query: {query}")
        st.write(f"Debug: Cleaned query: {cleaned_query}")
        st.write(f"Debug: Executing command: {search_command}")

        state = await self.manager.process_command_async(search_command)

        # Log the search results for debugging
        if state.search_context and state.search_context.results:
            st.write(f"Debug: Number of results: {len(state.search_context.results)}")
        else:
            st.write("Debug: No results found in search_context")

        # Log the state for debugging
        st.write(f"Debug: Search status: {state.status}")
        if state.error_message:
            st.error(f"Search error: {state.error_message}")

        if state.status == AgentStatus.SUCCESS:
            st.session_state.current_results = state
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)
            st.session_state.current_page = 1
        return state

    async def handle_pagination(self, direction: str):
        """Handle pagination commands"""
        command = "next" if direction == "next" else "prev"
        state = await self.manager.process_command_async(command)
        if state.status == AgentStatus.SUCCESS:
            st.session_state.current_results = state
            if direction == "next":
                st.session_state.current_page += 1
            else:
                st.session_state.current_page -= 1

    def render_search_interface(self):
        """Render search interface"""
        st.title("Talk2Competitors - Academic Paper Search")

        # Search box and filters in a form
        with st.form(key="search_form"):
            query = st.text_input("Search papers:", key="search_input")

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.text_input("Year (e.g., 2023):")
            with col2:
                citations = st.text_input("Min Citations:")
            with col3:
                sort_by = st.selectbox("Sort by:", ["Relevance", "Citations", "Year"])

            # Submit button
            submit_button = st.form_submit_button(label="Search")

            if submit_button and query:
                with st.spinner("Searching..."):
                    try:
                        state = asyncio.run(
                            self.process_search(query, year, citations, sort_by)
                        )
                        if state.status == AgentStatus.ERROR:
                            st.error(f"Search failed: {state.error_message}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    def render_results(self):
        """Render search results"""
        if st.session_state.current_results:
            state = st.session_state.current_results

            # Show search stats
            if state.search_context.total_results > 0:
                st.subheader(f"Found {state.search_context.total_results} papers")

                # Display papers
                if state.search_context.results:
                    for paper in state.search_context.results:
                        with st.container():
                            st.markdown(f"### {paper.title}")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                authors = ", ".join(
                                    a.get("name", "") for a in paper.authors[:3]
                                )
                                if len(paper.authors) > 3:
                                    authors += " et al."
                                st.write(f"**Authors:** {authors}")
                                if paper.abstract:
                                    with st.expander("Abstract"):
                                        st.write(paper.abstract)
                            with col2:
                                st.write(f"**Year:** {paper.year}")
                                st.write(f"**Citations:** {paper.citationCount}")
                            st.divider()

                    # Pagination
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        if st.session_state.current_page > 1:
                            if st.button("← Previous"):
                                asyncio.run(self.handle_pagination("prev"))
                    with col3:
                        if (
                            len(state.search_context.results) >= 10
                        ):  # Full page of results
                            if st.button("Next →"):
                                asyncio.run(self.handle_pagination("next"))
                    with col2:
                        st.write(f"Page {st.session_state.current_page}")
            else:
                st.info(
                    "No papers found matching your criteria. Try adjusting your search terms."
                )

    def render_history(self):
        """Render search history"""
        if st.session_state.search_history:
            with st.sidebar:
                st.subheader("Recent Searches")
                for query in reversed(st.session_state.search_history[-5:]):
                    if st.button(f"🔄 {query}", key=f"history_{query}"):
                        with st.spinner("Searching..."):
                            asyncio.run(self.process_search(query))

    def run(self):
        """Run the Streamlit app"""
        self.initialize_session_state()
        self.render_search_interface()
        self.render_results()
        self.render_history()


def main():
    st.set_page_config(
        page_title="Talk2Competitors - Paper Search", page_icon="📚", layout="wide"
    )
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
