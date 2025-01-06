**Project: Talk2Competitors - AI Agent Ecosystem**

**Overall Architecture:**

```markdown
- Main Agent
  - Agent_S2 (Semantic Scholar Agent)
    ✓ Search Papers [Current Focus]
    - Single Paper Reco
    - Multi Paper Reco
  - Agent_Zotero
    - Read Records
    - Plot Stats
    - Write Records
  - Agent_PDF
    - Ask Questions
  - Agent_Arxiv
    - Fetch PDF
```

**Current Focus - Agent_S2 Search Papers Component:**

1. Infrastructure:
   - Docker with Ollama for LLM backend
   - LangGraph for state management
   - Semantic Scholar API (pending integration)

**What We've Built:**

1. Core State Management System:

   - `agent_state.py`: Defines state models using Pydantic
   - `workflow_manager.py`: Handles LangGraph workflow
   - `test_state.py`: Testing infrastructure

2. Working Command Processing:
   ```python
   # Commands currently supported:
   - search [query]  # e.g., "search LLM papers"
   - help           # Shows available commands
   - [any]          # Generic command processing
   ```

**Recent Problem & Solution:**

1. Initial Problem:

   - LangGraph state management issues
   - Errors with state updates and transitions
   - Improper handling of node functions

2. Solutions Implemented:
   - Proper state conversion between LangGraph and AgentState
   - Complete state update handling in node functions
   - Robust error management and debugging
   - Message history preservation

**Code Structure:**

```python
# Core Components:
1. AgentState (agent_state.py)
   - Status tracking (IDLE/PROCESSING/SUCCESS/ERROR)
   - Message history
   - Search context
   - Error handling

2. WorkflowManager (workflow_manager.py)
   - Graph setup: start → process → finish
   - Command processing
   - State transitions
   - External interface

3. Testing (test_state.py)
   - Command processing tests
   - State transition verification
   - Search functionality testing
```

**Next Steps:**

1. Immediate Tasks:

   - Integrate Semantic Scholar API
   - Add paper metadata handling
   - Implement search result pagination

2. Future Components:

   - Paper recommendation system
   - Zotero integration
   - PDF processing capabilities
   - Arxiv integration

3. Long-term Goals:
   - Streamlit interface
   - Multi-paper analysis
   - Advanced search features
   - Cross-reference capabilities

**Technical Details to Remember:**

1. State Management:

   ```python
   # State must always include:
   {
       "status": AgentStatus,
       "current_step": str,
       "next_steps": List[str],
       "memory": ConversationMemory,
       "search_context": SearchContext,
       "error_message": Optional[str]
   }
   ```

2. LangGraph Integration:

   - Node functions must return complete state updates
   - Proper state conversion between formats
   - Maintain message history

3. Testing Requirements:
   - Command processing verification
   - State transition testing
   - Error handling validation

**Current Working State:**

```bash
# Example Output:
Debug: Processing command: search LLM papers
Status: AgentStatus.SUCCESS
Current step: finished
Messages:
- user: search LLM papers
- system: Search command received
```

**Next Immediate Focus:**

1. Semantic Scholar API Integration:

   - API client implementation
   - Response parsing
   - Result integration with state management

2. Enhanced Search Features:

   - Paper metadata extraction
   - Citation information
   - Author details
   - Publication dates

3. Error Handling Improvements:
   - API failure recovery
   - Rate limiting management
   - Data validation
