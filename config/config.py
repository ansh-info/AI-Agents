class Config:
    # LLM Configuration
    LLM_MODEL = "llama2:7b-chat"
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
Your capabilities include:
1. Searching for academic papers
2. Finding recommendations based on single papers
3. Finding recommendations based on multiple papers

You must:
1. Use the appropriate tool based on the task
2. Update the shared state with results
3. Format responses appropriately
4. Handle errors gracefully"""

    # Add other agent prompts as needed...


config = Config()
