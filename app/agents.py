import streamlit as st
from typing import Any, Dict, List

from agents.main_agent import main_agent
from config.config import config
from state.shared_state import shared_state


def initialize_page():
    """Initialize the Streamlit page with basic settings"""
    st.set_page_config(
        page_title="Research Paper Assistant", page_icon="📚", layout="wide"
    )
    st.title("Research Paper Assistant")
    st.markdown("---")


def display_paper(
    paper: Dict[str, Any], index: int, section: str = "main", show_save: bool = True
):
    """Display a single paper in an expander

    Args:
        paper: Paper data dictionary
        index: Index number for display
        section: Section identifier to create unique keys
        show_save: Whether to show save button
    """
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    year = paper.get("year", "N/A")
    abstract = paper.get("abstract", "No abstract available.")
    citations = paper.get("citationCount", 0)
    paper_id = paper.get("paperId", "")

    # Format authors
    author_names = [a.get("name", "") for a in authors if a]
    author_str = ", ".join(author_names[:3])
    if len(author_names) > 3:
        author_str += " et al."

    # Create expander for each paper
    with st.expander(f"{index}. {title} ({year})", expanded=False):
        st.markdown(f"**Authors:** {author_str}")
        st.markdown(f"**Year:** {year}")
        st.markdown(f"**Citations:** {citations}")
        st.markdown("**Abstract:**")
        st.markdown(abstract)
        st.markdown(f"**Paper ID:** {paper_id}")  # Debug: show paper ID

        col1, col2, col3 = st.columns(3)

        # Show recommendations button with unique key
        button_key = f"{section}_similar_{index}_{paper_id}"
        if col1.button(
            f"Get Similar Papers 🔍",
            key=button_key,
            help="Find papers similar to this one",
        ):
            if not paper_id:
                st.error("No paper ID available for recommendations")
            else:
                # Create a loading spinner
                with st.spinner("Finding similar papers..."):
                    try:
                        state = {
                            "message": f"Find papers similar to {paper_id}",
                            "response": None,
                            "error": None,
                        }
                        graph = main_agent.create_graph()
                        result = graph.invoke(state)

                        if result.get("error"):
                            st.error(
                                f"Error getting recommendations: {result['error']}"
                            )
                        elif not result.get("response"):
                            st.warning("No similar papers found")
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")

        # Show PDF link if available
        if pdf := paper.get("openAccessPdf"):
            pdf_url = pdf.get("url")
            if pdf_url:
                col2.markdown(f"[Open PDF 📄]({pdf_url})")

        # Add save to library button with unique key
        if show_save and col3.button(
            f"Save to Library 📚",
            key=f"{section}_save_{index}_{paper_id}",
            help="Save this paper to your library",
        ):
            st.info("Zotero integration coming soon!")


def display_papers_section():
    """Display the papers section with results from state"""
    papers = shared_state.get(config.StateKeys.PAPERS)
    if not papers:
        st.info("Search for papers to see results here.")
        return

    st.subheader(f"Found Papers ({len(papers)})")
    for i, paper in enumerate(papers, 1):
        display_paper(paper, i, section="search")


def display_query_section():
    """Display the query input section"""
    st.subheader("Search Papers")
    with st.form(key="search_form"):
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., recent papers about machine learning",
        )
        col1, col2, col3 = st.columns([2, 1, 1])
        submit_button = col1.form_submit_button("Search 🔍")

        if submit_button and query:
            # Create a loading spinner
            with st.spinner("Searching for papers..."):
                # Create initial state
                state = {
                    "message": query,
                    "response": None,
                    "error": None,
                }

                # Create and invoke graph
                graph = main_agent.create_graph()
                result = graph.invoke(state)

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.success("Search completed! View results below.")


def display_recommendations_section():
    """Display paper recommendations section"""
    if recommended_papers := shared_state.get(config.StateKeys.SELECTED_PAPERS):
        st.subheader(f"Recommended Papers ({len(recommended_papers)})")
        for i, paper in enumerate(recommended_papers, 1):
            display_paper(paper, i, section="recommendations", show_save=True)


def main():
    initialize_page()

    # Create main layout
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "History", "Settings"])

    if page == "Search":
        display_query_section()
        st.markdown("---")
        display_papers_section()
        st.markdown("---")
        display_recommendations_section()

    elif page == "History":
        st.subheader("Search History")
        if history := shared_state.get(config.StateKeys.CHAT_HISTORY):
            for entry in history:
                st.text(f"{entry.get('timestamp')}: {entry.get('content')}")
        else:
            st.info("No search history yet.")

    elif page == "Settings":
        st.subheader("Settings")
        st.info("Settings panel coming soon!")


if __name__ == "__main__":
    main()
