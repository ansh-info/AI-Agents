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
        """Initialize session state variables"""
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        if "current_results" not in st.session_state:
            st.session_state.current_results = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "selected_paper" not in st.session_state:
            st.session_state.selected_paper = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "paper_context" not in st.session_state:
            st.session_state.paper_context = {}

    def clean_search_query(self, query: str) -> str:
        """Clean and format the search query"""
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
        return " ".join(cleaned_query.split())

    async def process_search(
        self, query: str, year: str = None, citations: str = None, sort_by: str = None
    ):
        """Process search query and update state"""
        cleaned_query = self.clean_search_query(query)
        search_command = f"search {cleaned_query}"
        if year:
            search_command += f" year:{year}"
        if citations:
            search_command += f" citations>{citations}"
        if sort_by and sort_by != "Relevance":
            search_command += f" sort:{sort_by.lower()}"

        st.write(f"Debug: Original query: {query}")
        st.write(f"Debug: Cleaned query: {cleaned_query}")
        st.write(f"Debug: Executing command: {search_command}")

        state = await self.manager.process_command_async(search_command)

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

    async def process_paper_question(self, paper_id: str, question: str):
        """Process a question about a specific paper"""
        paper = st.session_state.paper_context.get(paper_id)
        if not paper:
            return "Paper not found in context."

        # Construct paper context for the LLM
        context = f"""
        Title: {paper.title}
        Authors: {', '.join(a.get('name', '') for a in paper.authors)}
        Year: {paper.year}
        Citations: {paper.citationCount}
        Abstract: {paper.abstract or 'Not available'}
        """

        # Use enhanced Ollama client to process the question
        try:
            response = await self.manager.llm_client.analyze_paper_question(
                context, question
            )
            return response
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def render_search_interface(self):
        """Render search interface"""
        st.title("Talk2Competitors - Academic Paper Search")

        with st.form(key="search_form"):
            query = st.text_input("Search papers:", key="search_input")

            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.text_input("Year (e.g., 2023):")
            with col2:
                citations = st.text_input("Min Citations:")
            with col3:
                sort_by = st.selectbox("Sort by:", ["Relevance", "Citations", "Year"])

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

    def render_paper_details(self, paper):
        """Render detailed paper view with chat interface"""
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"## {paper.title}")
        with col2:
            if st.button("← Back to Results"):
                st.session_state.selected_paper = None
                st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1:
            authors = ", ".join(a.get("name", "") for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            st.write(f"**Authors:** {authors}")
            if paper.abstract:
                st.write("**Abstract:**")
                st.write(paper.abstract)
        with col2:
            st.write(f"**Year:** {paper.year}")
            st.write(f"**Citations:** {paper.citationCount}")

        # Chat interface for the paper
        st.write("---")
        st.write("### Ask about this paper")

        # Display chat history for this paper
        if paper.paperId in st.session_state.chat_history:
            for msg in st.session_state.chat_history[paper.paperId]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Question input
        question = st.text_input(
            "Ask a question about this paper:", key=f"question_{paper.paperId}"
        )
        if st.button("Ask", key=f"ask_{paper.paperId}"):
            with st.spinner("Processing..."):
                response = asyncio.run(
                    self.process_paper_question(paper.paperId, question)
                )

                # Initialize chat history for this paper if needed
                if paper.paperId not in st.session_state.chat_history:
                    st.session_state.chat_history[paper.paperId] = []

                # Add to chat history
                st.session_state.chat_history[paper.paperId].extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                )

                st.rerun()

    def render_results(self):
        """Render search results with paper selection"""
        if st.session_state.current_results:
            state = st.session_state.current_results

            if state.search_context.total_results > 0:
                st.subheader(f"Found {state.search_context.total_results} papers")

                if state.search_context.results:
                    # Create tabs for results and selected paper
                    tabs = ["Search Results"]
                    if st.session_state.selected_paper:
                        tabs.append("Selected Paper")

                    current_tab = st.tabs(tabs)

                    with current_tab[0]:  # Search Results tab
                        for paper in state.search_context.results:
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"### {paper.title}")
                                with col2:
                                    if st.button(
                                        "Select", key=f"select_{paper.paperId}"
                                    ):
                                        st.session_state.selected_paper = paper.paperId
                                        st.experimental_rerun()

                                authors = ", ".join(
                                    a.get("name", "") for a in paper.authors[:3]
                                )
                                if len(paper.authors) > 3:
                                    authors += " et al."
                                st.write(f"**Authors:** {authors}")
                                st.write(
                                    f"**Year:** {paper.year} | **Citations:** {paper.citationCount}"
                                )
                                if paper.abstract:
                                    with st.expander("Abstract"):
                                        st.write(paper.abstract)
                                st.divider()

                        # Pagination
                        col1, col2, col3 = st.columns([2, 2, 2])
                        with col1:
                            if st.session_state.current_page > 1:
                                if st.button("← Previous"):
                                    asyncio.run(self.handle_pagination("prev"))
                        with col3:
                            if len(state.search_context.results) >= 10:
                                if st.button("Next →"):
                                    asyncio.run(self.handle_pagination("next"))
                        with col2:
                            st.write(f"Page {st.session_state.current_page}")

                    # Selected Paper tab
                    if len(tabs) > 1:
                        with current_tab[1]:
                            selected_paper = st.session_state.paper_context.get(
                                st.session_state.selected_paper
                            )
                            if selected_paper:
                                self.render_paper_details(selected_paper)
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
