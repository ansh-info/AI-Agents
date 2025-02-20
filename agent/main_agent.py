from typing import Any, Dict, List, TypedDict
import json
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
        try:
            # Get LLM's routing decision
            response = self.llm.get_response(
                system_prompt=config.MAIN_AGENT_PROMPT,
                user_input=query,
            )

            if not response:
                return {
                    "next_agent": None,
                    "query": query,
                    "response": "No response from routing agent",
                }

            # Clean up and fix the response
            response = response.strip()
            print(f"Processing response: {response}")

            try:
                # Find JSON boundaries
                first_brace = response.find("{")
                last_brace = response.find("}", first_brace)  # Find first closing brace

                if first_brace != -1 and last_brace != -1:
                    # Extract just the first complete JSON object
                    json_str = response[first_brace : last_brace + 1]
                    print(f"Attempting to parse JSON: {json_str}")

                    routing = json.loads(json_str)

                    if isinstance(routing, dict):
                        routing_type = routing.get("type", "")
                        agent_name = routing.get("agent")
                        confidence = float(routing.get("confidence", 0))
                        reason = routing.get("reason", "No reason provided")

                        print(
                            f"Parsed routing: type={routing_type}, agent={agent_name}, confidence={confidence}, reason={reason}"
                        )

                        if (
                            routing_type == "route"
                            and confidence >= 0.5
                            and agent_name in self.agents
                        ):
                            return {
                                "next_agent": agent_name,
                                "query": query,
                                "response": f"Routing to {agent_name} ({confidence:.2f} confidence): {reason}",
                            }

                return {
                    "next_agent": None,
                    "query": query,
                    "response": "Could not determine appropriate agent",
                }

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Failed to parse: {response}")
                return {
                    "next_agent": None,
                    "query": query,
                    "response": f"Invalid JSON format in routing response: {str(e)}",
                }

        except Exception as e:
            print(f"Error in determine_next_agent: {str(e)}")
            return {
                "next_agent": None,
                "query": query,
                "response": f"Error determining next agent: {str(e)}",
            }

    def route_to_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route the query to appropriate agent and handle response"""
        try:
            print("\nMain Agent Processing...")
            message = state.get("message", "")
            print(f"Received query: {message}")

            # Determine which agent should handle it
            routing = self.determine_next_agent(message)
            agent_name = routing["next_agent"]
            print(f"Selected agent: {agent_name}")

            if agent_name and agent_name in self.agents:
                # Update state with current agent
                shared_state.set(config.StateKeys.CURRENT_AGENT, agent_name)
                print(f"Routing to {agent_name}")

                # Create new state for sub-agent
                agent_state = {"message": routing["query"]}

                # Get the agent and its graph
                agent = self.agents[agent_name]
                graph = agent.create_graph()

                # Invoke the agent's graph
                sub_agent_result = graph.invoke(agent_state)
                print("Received response from sub-agent")

                # Process and format the response
                if sub_agent_result.get("response"):
                    state["response"] = sub_agent_result["response"]
                elif sub_agent_result.get("error"):
                    state["error"] = sub_agent_result["error"]
                    state["response"] = (
                        f"Error from {agent_name}: {sub_agent_result['error']}"
                    )
                else:
                    state["response"] = "No results found."

            else:
                state["response"] = "I'm not sure which agent should handle this query."
                state["error"] = "Could not determine appropriate agent"

            print("Main Agent finished processing")
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
