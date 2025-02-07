from typing import Any, Dict, List, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Type definition for agent state"""

    message: str
    response: str | None
    error: str | None


from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config.config import config
from state.shared_state import shared_state
from agents.s2_agent import s2_agent
from utils.llm import llm_manager


class MainAgent:
    def __init__(self):
        self.llm = llm_manager
        # Map of agent names to agent instances
        self.agents = {
            config.AgentNames.S2: s2_agent,
            # We'll add other agents later
            # config.AgentNames.ZOTERO: zotero_agent,
            # config.AgentNames.PDF: pdf_agent,
            # config.AgentNames.ARXIV: arxiv_agent
        }

        # Create prompt template with routing examples
        system_prompt = (
            config.MAIN_AGENT_PROMPT
            + """

Here are some examples of how to route queries:

Example 1: Paper Search
Human: Find papers about machine learning
Assistant: This query requires searching for academic papers. I'll route this to the Semantic Scholar agent.
Action: route_to_agent(agent="semantic_scholar_agent", query="Find papers about machine learning")

Example 2: Multiple Agent Workflow
Human: Find papers about transformers and save them to my Zotero library
Assistant: This requires multiple steps:
1. First, we'll search for papers using the Semantic Scholar agent
2. Then, we'll save the results using the Zotero agent
Let's start with the search.
Action: route_to_agent(agent="semantic_scholar_agent", query="Find papers about transformers")

Example 3: PDF Analysis
Human: Can you analyze this PDF and answer questions about it?
Assistant: This requires PDF processing and RAG operations. I'll route this to the PDF agent.
Action: route_to_agent(agent="pdf_agent", query="Analyze the provided PDF")

Remember to:
1. Choose the most appropriate agent for each task
2. Break down multi-step tasks into proper sequences
3. Maintain context between agent calls
4. Handle errors and responses appropriately"""
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{query}")]
        )

    def determine_next_agent(self, query: str) -> Dict[str, Any]:
        """
        Determine which agent should handle the query

        Args:
            query: User's query

        Returns:
            Dict with next agent and possibly modified query
        """
        messages = [HumanMessage(content=query)]
        context = shared_state.get_current_context()

        # Get LLM's routing decision
        response = self.llm.get_response(
            system_prompt=config.MAIN_AGENT_PROMPT,
            user_input=query,
            additional_context=context,
        )

        # Map of keywords to agent names
        agent_keywords = {
            config.AgentNames.S2: [
                "paper",
                "search",
                "find",
                "semantic scholar",
                "papers",
                "research",
                "publication",
            ],
            config.AgentNames.ZOTERO: ["zotero", "save", "library", "reference"],
            config.AgentNames.PDF: ["pdf", "read", "analyze", "content"],
            config.AgentNames.ARXIV: ["arxiv", "download", "get pdf"],
        }

        # Check response and query against keywords
        response_lower = response.lower()
        query_lower = query.lower()

        for agent_name, keywords in agent_keywords.items():
            # Check if any keyword is in either response or query
            if any(keyword in response_lower for keyword in keywords) or any(
                keyword in query_lower for keyword in keywords
            ):
                return {
                    "next_agent": agent_name,
                    "query": query,
                    "response": f"Routing to {agent_name} to handle this query.",
                }

        return {
            "next_agent": None,
            "query": query,
            "response": "I'm not sure which agent should handle this query.",
        }

    def route_to_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route the query to appropriate agent"""
        try:
            # Get current message
            message = state.get("message", "")

            # Determine which agent should handle it
            routing = self.determine_next_agent(message)
            agent_name = routing["next_agent"]

            if agent_name and agent_name in self.agents:
                # Update state with current agent
                shared_state.set(config.StateKeys.CURRENT_AGENT, agent_name)

                # Create new state for sub-agent
                agent_state = {"message": routing["query"]}

                # Get the agent and its graph
                agent = self.agents[agent_name]
                graph = agent.create_graph()

                # Invoke the agent's graph
                result = graph.invoke(agent_state)

                # Update main state with result
                state["response"] = result.get("response")
                state["error"] = result.get("error")

            else:
                state["response"] = routing["response"]
                state["error"] = "Could not determine appropriate agent"

            return state

        except Exception as e:
            error_msg = f"Error in main agent: {str(e)}"
            state["error"] = error_msg
            shared_state.set(config.StateKeys.ERROR, error_msg)
            return state

    def create_graph(self) -> StateGraph:
        """Create the main workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("route_to_agent", self.route_to_agent)

        # Add edges
        workflow.add_edge("route_to_agent", END)

        # Set entry point
        workflow.set_entry_point("route_to_agent")

        return workflow.compile()


# Create a global instance
main_agent = MainAgent()
