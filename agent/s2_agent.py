from typing import List, Dict, Any, Type, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import s2_tools
from utils.llm import llm_manager


class S2AgentState(TypedDict):
    """Type definition for S2 agent state"""

    message: str
    response: str | None
    error: str | None


class SemanticScholarAgent:
    def __init__(self):
        try:
            self.llm = llm_manager.llm.bind_tools(s2_tools)
            self.tool_executor = ToolExecutor(s2_tools)

            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", config.S2_AGENT_PROMPT),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )

            # Create the chain
            self.chain = (
                RunnablePassthrough.assign(history=lambda x: [])
                | self.prompt
                | self.llm
            )
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            message = state.get("message", "")
            context = shared_state.get_current_context()

            if context:
                message += f"\n\nContext:\n{context}"

            try:
                # Get initial response with potential tool calls
                response = self.chain.invoke({"input": message})
            except ConnectionError as conn_err:
                state["error"] = (
                    f"Connection error: Please ensure Ollama service is running. Error: {str(conn_err)}"
                )
                return state
            except Exception as chain_err:
                state["error"] = f"Chain error: {str(chain_err)}"
                return state

            messages = [HumanMessage(content=message), response]

            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        tool_output = self.tool_executor.invoke(
                            tool_call.name, tool_call.args
                        )
                        messages.append(
                            ToolMessage(
                                content=str(tool_output), tool_call_id=tool_call.id
                            )
                        )
                    except Exception as tool_error:
                        messages.append(
                            ToolMessage(
                                content=str(
                                    {"error": f"Tool error: {str(tool_error)}"}
                                ),
                                tool_call_id=tool_call.id,
                            )
                        )

                try:
                    final_response = self.llm.invoke(messages)
                    response_content = final_response.content
                except Exception as llm_err:
                    state["error"] = f"LLM response error: {str(llm_err)}"
                    return state
            else:
                response_content = response.content

            state["response"] = response_content
            shared_state.add_to_chat_history("assistant", response_content)
            return state

        except Exception as e:
            state["error"] = f"Error in S2 agent: {str(e)}"
            shared_state.set(config.StateKeys.ERROR, state["error"])
            return state

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph(S2AgentState)
        workflow.add_node("process_message", self.handle_message)
        workflow.add_edge("process_message", END)
        workflow.set_entry_point("process_message")
        return workflow.compile()


# Create a global instance
s2_agent = SemanticScholarAgent()
