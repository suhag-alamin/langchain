from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
)

st.header("ChatBot")

# chatbot
# chat history with session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(
            content="You are a AI assistant. Always answer in short."
        )
    ]

# display previous message
for msg in st.session_state.chat_history[1:]:
    # compare 2 and if true assign assistant and if false assign user
    role = "assistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.write(msg.content)

# print chat history
with st.sidebar:
    st.subheader("Chat History")
    if not st.session_state.chat_history[1:]:
        st.write("Empty")
    else:
        for msg in st.session_state.chat_history:
            role = "AI" if isinstance(msg, AIMessage) else "Human"
            st.write(f"**{role}:** {msg.content}")


# take user input

user_input = st.chat_input("Type your message")

if user_input:
    if user_input.strip().lower() == "exit":
        st.stop()

    # store the user message
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    # display the user message:
    with st.chat_message("user"):
        st.write(user_input)

    result = model.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=result.content))

    with st.chat_message("assistant"):
        st.write(result.content)
