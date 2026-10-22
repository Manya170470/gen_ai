from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

st.title("🤖 GoogleChatBot – AI QnA Bot")
st.markdown("My QnA Bot with Google Search and Groq!")

# Initialize model, search and memory
model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt=(
        "You are a helpful assistant who can search "
        "Google to answer questions when needed."
    )
)

# Display previous messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from Streamlit
if query := st.chat_input("Ask anything..."):
    # Display user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "11"}}
            )
            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )