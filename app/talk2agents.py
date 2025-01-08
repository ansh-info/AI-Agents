import asyncio
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from agent.enhanced_workflow import EnhancedWorkflowManager
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentStatus


class SearchAgent:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.s2_client = SemanticScholarClient()
        self.search_limit = 10

    async def enhance_query(self, query: str) -> str:
        """Use LLM to enhance the search query"""
        prompt = f"""Enhance this search query for academic papers: "{query}"
        Add relevant academic terms and remove non-essential words.
        Return only the enhanced query."""

        try:
            enhanced = await self.ollama_client.generate(prompt=prompt, max_tokens=50)
            return enhanced.strip() or query
        except Exception as e:
            print(f"Query enhancement failed: {str(e)}")
            return query

    async def analyze_paper_question(self, paper_context: str, question: str) -> str:
        """Answer questions about a paper"""
        prompt = f"""Based on this paper information:
        {paper_context}
        
        Answer this question: {question}
        
        Provide a clear, concise answer based only on the available information.
        If the information isn't in the context, say so."""

        try:
            response = await self.ollama_client.generate(prompt=prompt, max_tokens=300)
            return response.strip()
        except Exception as e:
            return f"Error analyzing paper: {str(e)}"

    async def search_papers(self, query: str, filters: dict = None) -> tuple:
        """Search for papers with the given query and filters"""
        try:
            # Enhance query
            enhanced_query = await self.enhance_query(query)

            # Prepare search parameters
            params = {"query": enhanced_query, "limit": self.search_limit, "offset": 0}

            # Add filters if provided
            if filters:
                if filters.get("year"):
                    params["year"] = filters["year"]
                if filters.get("citations"):
                    params["min_citations"] = filters["citations"]

            # Perform search
            results = await self.s2_client.search_papers(**params)
            return results, None

        except Exception as e:
            return None, str(e)


class StreamlitApp:
    def __init__(self):
        self.agent = SearchAgent()

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
            st.session_state.chat_history = {}
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

    async def process_search(self, query: str, year: str = None, citations: str = None):
        """Process search query and update state"""
        cleaned_query = self.clean_search_query(query)

        # Prepare filters
        filters = {}
        if year:
            try:
                filters["year"] = int(year)
            except ValueError:
                st.error("Invalid year format")
                return
        if citations:
            try:
                filters["citations"] = int(citations)
            except ValueError:
                st.error("Invalid citations format")
                return

        # Execute search
        results, error = await self.agent.search_papers(cleaned_query, filters)

        if error:
            st.error(f"Search failed: {error}")
            return

        if results:
            st.session_state.current_results = results
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)

            # Store papers in context
            for paper in results.papers:
                st.session_state.paper_context[paper.paperId] = paper

    async def process_paper_question(self, paper_id: str, question: str):
        """Process a question about a specific paper"""
        paper = st.session_state.paper_context.get(paper_id)
        if not paper:
            return "Paper not found in context."

        # Construct paper context
        context = f"""
        Title: {paper.title}
        Authors: {', '.join(a.get('name', '') for a in paper.authors)}
        Year: {paper.year}
        Citations: {paper.citationCount}
        Abstract: {paper.abstract or 'Not available'}
        """

        return await self.agent.analyze_paper_question(context, question)

    def render_search_interface(self):
        """Render search interface"""
        st.title("Academic Paper Search Assistant")

        with st.form(key="search_form"):
            query = st.text_input("Search papers:", key="search_input")

            col1, col2 = st.columns(2)
            with col1:
                year = st.text_input("Year (e.g., 2023):")
            with col2:
                citations = st.text_input("Min Citations:")

            submit_button = st.form_submit_button(label="Search")

            if submit_button and query:
                with st.spinner("Searching..."):
                    try:
                        asyncio.run(self.process_search(query, year, citations))
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

        st.write(f"**Authors:** {', '.join(a.get('name', '') for a in paper.authors)}")
        st.write(f"**Year:** {paper.year} | **Citations:** {paper.citationCount}")

        if paper.abstract:
            st.write("**Abstract:**")
            st.write(paper.abstract)

        # Chat interface
        st.write("---")
        st.write("### Ask about this paper")

        # Display chat history
        if paper.paperId in st.session_state.chat_history:
            for msg in st.session_state.chat_history[paper.paperId]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Question input
        question = st.text_input("Ask a question:", key=f"question_{paper.paperId}")
        if st.button("Ask", key=f"ask_{paper.paperId}"):
            if question:
                with st.spinner("Thinking..."):
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
            results = st.session_state.current_results

            if results.papers:
                st.subheader(f"Found {results.total} papers")

                if st.session_state.selected_paper:
                    # Show selected paper
                    paper = st.session_state.paper_context.get(
                        st.session_state.selected_paper
                    )
                    if paper:
                        self.render_paper_details(paper)
                else:
                    # Show search results
                    for paper in results.papers:
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"### {paper.title}")
                            with col2:
                                if st.button("Select", key=f"select_{paper.paperId}"):
                                    st.session_state.selected_paper = paper.paperId
                                    st.rerun()

                            st.write(
                                f"**Authors:** {', '.join(a.get('name', '') for a in paper.authors[:3])}"
                            )
                            st.write(
                                f"**Year:** {paper.year} | **Citations:** {paper.citationCount}"
                            )
                            if paper.abstract:
                                with st.expander("Abstract"):
                                    st.write(paper.abstract)
                            st.divider()
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
        page_title="Academic Paper Search", page_icon="📚", layout="wide"
    )
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
