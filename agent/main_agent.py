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
                print("Empty response from LLM")
                return {
                    "next_agent": None,
                    "query": query,
                    "response": "No response from routing agent",
                }

            # Clean up the response
            response = response.strip()
            print(f"Processing response: {response}")

            try:
                # Find JSON boundaries if needed
                first_brace = response.find("{")
                last_brace = response.rfind("}")
                if first_brace != -1 and last_brace != -1:
                    json_str = response[first_brace : last_brace + 1]
                else:
                    json_str = response

                print(f"Attempting to parse JSON: {json_str}")
                routing = json.loads(json_str)

                # Validate the routing object
                if not isinstance(routing, dict):
                    return {
                        "next_agent": None,
                        "query": query,
                        "response": "Invalid routing format - not a dictionary",
                    }

                # Validate required fields
                required_fields = ["type", "agent", "confidence", "reason"]
                if not all(field in routing for field in required_fields):
                    return {
                        "next_agent": None,
                        "query": query,
                        "response": f"Invalid routing format - missing fields. Found: {list(routing.keys())}",
                    }

                # Extract and validate fields
                routing_type = routing["type"]
                agent_name = routing["agent"]
                try:
                    confidence = float(routing["confidence"])
                except ValueError:
                    confidence = 0.0
                reason = routing["reason"]

                print(
                    f"Parsed routing: type={routing_type}, agent={agent_name}, confidence={confidence}, reason={reason}"
                )

                # Validate routing type
                if routing_type != "route":
                    return {
                        "next_agent": None,
                        "query": query,
                        "response": f"Invalid routing type: {routing_type}",
                    }

                # Check confidence and agent validity
                if confidence >= 0.5 and agent_name in self.agents:
                    return {
                        "next_agent": agent_name,
                        "query": query,
                        "response": f"Routing to {agent_name} ({confidence:.2f} confidence): {reason}",
                    }
                else:
                    return {
                        "next_agent": None,
                        "query": query,
                        "response": (
                            "Low confidence" if confidence < 0.5 else "Invalid agent"
                        )
                        + f": {reason}",
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
                    state["response"] = (
                        f"Results from {agent_name}:\n\n{sub_agent_result['response']}"
                    )
                else:
                    state["response"] = "No results found."

                if sub_agent_result.get("error"):
                    state["error"] = sub_agent_result["error"]

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
