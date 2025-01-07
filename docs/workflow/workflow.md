The theoretical setup and workflow step by step.

1. **Overall Architecture**

   - Main Agent acts as an orchestrator
   - Sub-agents (S2, Zotero, PDF, Arxiv) handle specialized tasks
   - LangGraph manages state transitions and information flow
   - Streamlit provides the user interface

2. **LangGraph State Management**

   - States are like memory containers storing:
     - Current conversation context
     - Search results
     - Selected papers
     - Agent status
   - Each interaction updates these states
   - States persist throughout the conversation
   - Other agents can access this state information

3. **Workflow Steps**

   ```
   User Query -> Main Agent -> Sub-Agent -> State Update -> Response -> UI
   ```

   For example:

   - User: "Find papers about LLMs"
   - Main Agent: Understands intent, routes to Agent_S2
   - Agent_S2: Calls Semantic Scholar API
   - State: Stores search results
   - UI: Displays paper table
   - User: "Tell me about row 4"
   - Main Agent: Uses state to access row 4 data
   - Response: Provides paper details

4. **Agent Communication**

   - Main Agent orchestrates sub-agents
   - Each sub-agent has specific responsibilities:
     - Agent_S2: Paper search and recommendations
     - Agent_Zotero: Bibliography management
     - Agent_PDF: Document analysis
     - Agent_Arxiv: PDF retrieval
   - Agents communicate through state updates

5. **Docker Setup**

   ```
   Containers:
   - Ollama (LLM service)
   - Streamlit (UI)
   - Agents (Python service)
   - State Management (Shared volume)
   ```

6. **Development Phases**
   Phase 1: Basic Setup

   - Implement Main Agent
   - Setup Agent_S2
   - Basic state management
   - Simple UI

   Phase 2: Search Implementation

   - API integration
   - Result processing
   - State storage
   - Table display

   Phase 3: Context Management

   - Row selection
   - Paper details
   - Conversation memory
