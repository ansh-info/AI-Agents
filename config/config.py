class Config:
    # LLM Configuration
    LLM_MODEL = "llama3.2:1b-instruct-q3_K_M"
    TEMPERATURE = 0.7

    # API Endpoints
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    # State Keys
    class StateKeys:
        PAPERS = "papers"
        SELECTED_PAPERS = "selected_papers"
        CURRENT_TOOL = "current_tool"
        CURRENT_AGENT = "current_agent"
        RESPONSE = "response"
        ERROR = "error"
        CHAT_HISTORY = "chat_history"
        USER_INFO = "user_info"
        MEMORY = "memory"

    # Agent Names
    class AgentNames:
        MAIN = "main_agent"
        S2 = "semantic_scholar_agent"
        ZOTERO = "zotero_agent"
        PDF = "pdf_agent"
        ARXIV = "arxiv_agent"

    # Tool Names
    class ToolNames:
        # S2 Tools
        S2_SEARCH = "search_papers"
        S2_SINGLE_REC = "single_paper_recommendation"
        S2_MULTI_REC = "multi_paper_recommendation"

        # Zotero Tools
        ZOTERO_READ = "zotero_read"
        ZOTERO_WRITE = "zotero_write"

        # PDF Tools
        PDF_RAG = "pdf_rag"

        # arXiv Tools
        ARXIV_DOWNLOAD = "arxiv_download"

    # System Prompts
    MAIN_AGENT_PROMPT = """You are a supervisory AI agent responsible for managing multiple specialized sub-agents.
Your role is to:
1. Analyze user queries
2. Determine which sub-agent is most appropriate for the task
3. Route the query to the chosen sub-agent
4. Manage the workflow between agents

Available agents and their capabilities:
- Semantic Scholar Agent (s2): Search papers, get paper recommendations
- Zotero Agent: Read from and write to Zotero library
- PDF Agent: RAG operations on PDFs
- arXiv Agent: Download PDFs from arXiv

You must always:
1. Choose the most appropriate agent for the task
2. Maintain and update the shared state
3. Coordinate between agents when needed
4. Ensure proper error handling"""

    S2_AGENT_PROMPT = """You are a specialized agent for interacting with Semantic Scholar.
Your role is to help users find and manage academic papers. You have access to several tools:

1. search_papers: Search for academic papers using keywords
2. get_single_paper_recommendations: Get recommendations based on a single paper
3. get_multi_paper_recommendations: Get recommendations based on multiple papers

You should:
1. Use the appropriate tool based on the user's needs
2. Search first before getting recommendations
3. Format responses clearly and concisely
4. Handle errors gracefully

Here are some examples of how to use the tools:

Example 1: Basic Search
Human: Find papers about transformer architecture
Assistant: I'll search for papers about transformer architecture.
Tool: search_papers(query="transformer architecture neural networks", limit=5)
Tool Result: {"status": "success", "papers": [...], "total": 5}
Assistant: I found several papers about transformer architecture. Here are the key ones: [list papers]

Example 2: Paper Recommendations
Human: Can you find papers similar to the one about attention is all you need?
Assistant: I'll get recommendations based on that paper.
Tool: search_papers(query="attention is all you need transformer", limit=1)
Tool Result: {"status": "success", "papers": [{"paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776", ...}]}
Tool: get_single_paper_recommendations(paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776", limit=5)
Tool Result: {"status": "success", "recommendations": [...]}
Assistant: Here are some similar papers to "Attention Is All You Need": [list recommendations]

Remember to:
- Always search first if you don't have a paper ID
- Use clear, specific search queries
- Provide context from paper abstracts when relevant
- Handle multi-step queries appropriately"""

    # Add other agent prompts as needed...


config = Config()
