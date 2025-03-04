"""
This package contains tools for agent s2
"""

from tools.s2.search import search_papers
from tools.s2.single_paper_rec import get_single_paper_recommendations
from tools.s2.multi_paper_rec import get_multi_paper_recommendations

# Export all tools in a list for easy access
s2_tools = [
    search_papers,
    get_single_paper_recommendations,
    get_multi_paper_recommendations,
]

__all__ = [
    "search_papers",
    "get_single_paper_recommendations",
    "get_multi_paper_recommendations",
    "s2_tools",
]
