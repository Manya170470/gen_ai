from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from uuid import uuid4

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver


# -------------------- Streamlit UI --------------------

st.title("🤖 GoogleChatBot – AI QnA Bot")
st.markdown("My QnA Bot with Google Search and Groq!")


# -------------------- Initialize Session State --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep the same checkpointer across Streamlit reruns
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# Give each browser session its own conversation thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())


# -------------------- Initialize Model, Search and Agent --------------------

model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=st.session_state.memory,
    system_prompt=(
        "You are a helpful AI assistant who can search Google "
        "to answer questions when needed. "
        "Use the previous messages in the conversation to understand "
        "follow-up questions and references such as 'he', 'she', 'it', "
        "or 'his relationship'. Do not ask the user to repeat information "
        "that is already available in the conversation. "
        "If the question is clear from the conversation, answer directly."
    )
)


# -------------------- Display Previous Messages --------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------- Get User Input --------------------

if query := st.chat_input("Ask anything..."):

    # Display and store the user's message
    st.chat_message("user").markdown(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": query}
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                answer = response["messages"][-1].content
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")