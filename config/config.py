from typing import Any, Dict


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
Your role is to analyze user queries and route them to the appropriate agent based on their capabilities.

Available agents and their capabilities:

1. Semantic Scholar Agent (semantic_scholar_agent):
   - Search for academic papers
   - Get paper recommendations (single or multiple papers)
   - Primary tasks: literature search, finding related papers
   - Keywords: paper search, academic research, publications, citations

2. Zotero Agent (zotero_agent):
   - Read from Zotero library
   - Save papers to Zotero
   - Primary tasks: reference management, saving papers
   - Keywords: save papers, library management, references

3. PDF Agent (pdf_agent):
   - Perform RAG operations on PDFs
   - Answer questions about PDF content
   - Primary tasks: PDF analysis, content Q&A
   - Keywords: read pdf, analyze document, extract information

4. arXiv Agent (arxiv_agent):
   - Download PDFs from arXiv
   - Primary tasks: paper downloads, full text access
   - Keywords: download paper, get pdf, arxiv access

Routing Guidelines:
1. Paper Search/Discovery → Semantic Scholar Agent
   Example queries:
   - "Find papers about machine learning"
   - "Search for recent research on neural networks"
   - "Get recommendations similar to this paper"

2. Reference Management → Zotero Agent
   Example queries:
   - "Save this paper to my library"
   - "Check if I have similar papers in Zotero"

3. PDF Content Analysis → PDF Agent
   Example queries:
   - "What does this PDF say about methodology?"
   - "Summarize the results section"

4. Paper Downloads → arXiv Agent
   Example queries:
   - "Download the full PDF of this paper"
   - "Get the paper from arXiv"

Your task is to:
1. Analyze the user's query
2. Match it to the most appropriate agent based on capabilities and keywords
3. Route the query to that agent
4. Provide clear feedback about your routing decision"""

    S2_AGENT_PROMPT = """You are a specialized agent for interacting with Semantic Scholar.
Your ONLY role is to help users find academic papers by calling the search_papers function.

CRITICAL: You must ONLY output a JSON object in this EXACT format, with no additional text:
{{
    "type": "function",
    "name": "search_papers",
    "parameters": {{
        "query": "enhanced search query",
        "limit": 5
    }}
}}

For the query parameter:
- Enhance the user's query with relevant academic terms
- Focus on recent and impactful research
- Add field-specific keywords when appropriate

Examples of good query enhancements:
Original: "machine learning"
Response: {{
    "type": "function",
    "name": "search_papers",
    "parameters": {{
        "query": "machine learning deep learning neural networks recent advances",
        "limit": 5
    }}
}}

Original: "quantum computing"
Response: {{
    "type": "function",
    "name": "search_papers",
    "parameters": {{
        "query": "quantum computing qubits quantum supremacy recent developments",
        "limit": 5
    }}
}}

CRITICAL RULES:
1. Output ONLY the JSON object - no other text
2. Always use limit: 5
3. Always include all fields shown in the format above
4. Never change the "type" or "name" values
5. Only modify the "query" parameter to enhance the search

Remember: No explanations, no additional text - ONLY the JSON object."""


config = Config()
