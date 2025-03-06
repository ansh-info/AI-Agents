#!/usr/bin/env python3

"""
Talk2Papers: A Streamlit app for academic paper search and recommendations
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import streamlit as st
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.main_agent import get_app

# Page configuration
st.set_page_config(page_title="Talk2Papers", page_icon="📚", layout="wide")

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Welcome to Talk2Papers!"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)

if "app" not in st.session_state:
    st.session_state.app = get_app(st.session_state.unique_id)

# Get the app
app = st.session_state.app

# Check OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Main layout with two columns
main_col1, main_col2 = st.columns([2, 8])

# First column (Settings)
with main_col1:
    with st.container(border=True):
        st.write(
            """
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            📚 Talk2Papers
            </h3>
            """,
            unsafe_allow_html=True,
        )

        # LLM Selection
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        llm_option = st.selectbox(
            "Select LLM Model", llms, index=0, key="st_selectbox_llm"
        )

# Second column (Chat Interface)
with main_col2:
    # Chat history container
    chat_container = st.container(border=True)
    with chat_container:
        st.write("#### 💬 Chat History")

        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(
                message["content"].role,
                avatar="🤖" if message["content"].role != "user" else "👤",
            ):
                st.markdown(message["content"].content)
                st.empty()

    # Input at the bottom
    prompt = st.chat_input("Search for papers or ask questions...", key="st_chat_input")

    if prompt:
        # Display user message
        prompt_msg = ChatMessage(prompt, role="user")
        st.session_state.messages.append({"type": "message", "content": prompt_msg})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Processing your request..."):
                # Get chat history
                history = [
                    (m["content"].role, m["content"].content)
                    for m in st.session_state.messages
                    if m["type"] == "message"
                ]

                # Convert to ChatMessage objects
                chat_history = [
                    (
                        SystemMessage(content=m[1])
                        if m[0] == "system"
                        else (
                            HumanMessage(content=m[1])
                            if m[0] == "human"
                            else AIMessage(content=m[1])
                        )
                    )
                    for m in history
                ]

                # Create config
                config = {"configurable": {"thread_id": st.session_state.unique_id}}

                # Set LLM model
                os.environ["AIAGENTS4PHARMA_LLM_MODEL"] = llm_option

                # Get response from agent
                response = app.invoke(
                    {"messages": [HumanMessage(content=prompt)]}, config=config
                )

                # Add response to chat history
                assistant_msg = ChatMessage(
                    response["messages"][-1].content, role="assistant"
                )
                st.session_state.messages.append(
                    {"type": "message", "content": assistant_msg}
                )

                # Display response
                st.markdown(response["messages"][-1].content)
                st.empty()
