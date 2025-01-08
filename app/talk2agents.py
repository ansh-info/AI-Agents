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

    async def process_search(self, query: str):
        """Process search query and update state"""
        state = await self.manager.process_command_async(f"search {query}")
        if state.status == AgentStatus.SUCCESS:
            st.session_state.current_results = state
            st.session_state.search_history.append(query)

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

        # Search box
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Search papers:", key="search_input")
        with col2:
            if st.button("Search"):
                if query:
                    asyncio.run(self.process_search(query))

        # Filters
        with st.expander("Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.text_input("Year (e.g., 2023):")
            with col2:
                citations = st.text_input("Min Citations:")
            with col3:
                sort_by = st.selectbox("Sort by:", ["Relevance", "Citations", "Year"])

    def render_results(self):
        """Render search results"""
        if st.session_state.current_results:
            state = st.session_state.current_results

            # Show search stats
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
                        len(state.search_context.results) == 10
                    ):  # Assuming 10 results per page
                        if st.button("Next →"):
                            asyncio.run(self.handle_pagination("next"))
                with col2:
                    st.write(f"Page {st.session_state.current_page}")

    def render_history(self):
        """Render search history"""
        if st.session_state.search_history:
            with st.sidebar:
                st.subheader("Search History")
                for query in reversed(st.session_state.search_history[-5:]):
                    if st.button(f"🔄 {query}", key=f"history_{query}"):
                        asyncio.run(self.process_search(query))

    def run(self):
        """Run the Streamlit app"""
        self.initialize_session_state()
        self.render_search_interface()
        self.render_results()
        self.render_history()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
