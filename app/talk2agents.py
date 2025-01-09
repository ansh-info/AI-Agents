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
        # Create a placeholder for status messages
        status_placeholder = st.empty()

        # Show initial status
        status_placeholder.info("🔍 Processing search query...")

        cleaned_query = self.clean_search_query(query)
        status_placeholder.info(f"🧹 Cleaned query: '{cleaned_query}'")

        # Prepare filters
        filters = {}
        filter_text = []
        if year:
            try:
                filters["year"] = int(year)
                filter_text.append(f"Year: {year}")
            except ValueError:
                st.error("❌ Invalid year format")
                return
        if citations:
            try:
                filters["citations"] = int(citations)
                filter_text.append(f"Min Citations: {citations}")
            except ValueError:
                st.error("❌ Invalid citations format")
                return

        if filter_text:
            status_placeholder.info(f"⚙️ Applying filters: {', '.join(filter_text)}")

        # Execute search with progress updates
        status_placeholder.info("🤖 Enhancing query using AI...")
        results, error = await self.agent.search_papers(cleaned_query, filters)

        if error:
            status_placeholder.error(f"❌ Search failed: {error}")
            return

        if results:
            # Update status with success message
            status_placeholder.success(f"✅ Found {results.total} papers!")

            st.session_state.current_results = results
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)

            # Store papers in context
            for paper in results.papers:
                st.session_state.paper_context[paper.paperId] = paper

            # Clear status after a short delay
            await asyncio.sleep(2)
            status_placeholder.empty()

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
        # Custom CSS
        st.markdown(
            """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .search-container {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Header
        st.markdown(
            """
            <h1 style='text-align: center; color: #1e88e5; margin-bottom: 1rem;'>
                📚 Academic Paper Search Assistant
            </h1>
            <p style='text-align: center; color: #666; margin-bottom: 2rem;'>
                Search through academic papers, explore research, and ask questions about papers
            </p>
        """,
            unsafe_allow_html=True,
        )

        # Debug section for showing backend processes
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []

        # Search interface
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        with st.form(key="search_form"):
            query = st.text_input(
                "🔍 Search papers:",
                key="search_input",
                placeholder="Enter your research query (e.g., 'machine learning in healthcare')",
            )

            cols = st.columns([2, 2, 1])
            with cols[0]:
                year = st.text_input("📅 Year:", placeholder="e.g., 2023")
            with cols[1]:
                citations = st.text_input("📊 Min Citations:", placeholder="e.g., 100")
            with cols[2]:
                search_button = st.form_submit_button(
                    "🔍 Search", use_container_width=True, type="primary"
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Process search
        if search_button and query:
            # Create tabs for Results and Debug Output
            results_tab, debug_tab = st.tabs(["📊 Search Results", "🔧 Debug Output"])

            with debug_tab:
                st.markdown("### 🔍 Search Process")
                debug_container = st.empty()

            with results_tab:
                with st.spinner("🔍 Searching..."):
                    try:
                        asyncio.run(self.process_search(query, year, citations))
                    except Exception as e:
                        st.error(f"⚠️ Search failed: {str(e)}")

            # Update debug output in the debug tab
            with debug_container:
                for msg in st.session_state.debug_messages:
                    st.write(msg)

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
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 1rem; background-color: #e3f2fd; 
                    border-radius: 10px; margin-bottom: 2rem;'>
                        <h2 style='color: #1e88e5; margin: 0;'>
                            📚 Found {results.total} papers
                        </h2>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

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
                            st.markdown(
                                f"""
                                <div style='background-color: white; padding: 1.5rem; 
                                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                                margin-bottom: 1rem;'>
                                    <h3 style='color: #1e88e5; margin-top: 0;'>{paper.title}</h3>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            col1, col2 = st.columns([4, 1])
                            with col1:
                                authors = ", ".join(
                                    a.get("name", "") for a in paper.authors[:3]
                                )
                                if len(paper.authors) > 3:
                                    authors += " et al."
                                st.write(f"👥 **Authors:** {authors}")
                                st.write(
                                    f"📅 **Year:** {paper.year} | 📊 **Citations:** {paper.citationCount}"
                                )
                                if paper.abstract:
                                    with st.expander("📖 Show Abstract"):
                                        st.write(paper.abstract)
                            with col2:
                                if st.button(
                                    "🔍 View Details",
                                    key=f"select_{paper.paperId}",
                                    use_container_width=True,
                                ):
                                    st.session_state.selected_paper = paper.paperId
                                    st.rerun()
            else:
                st.warning(
                    "🔍 No papers found matching your criteria. Try adjusting your search terms."
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
