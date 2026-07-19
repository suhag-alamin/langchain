from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # model="openai/gpt-oss-120b",
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    # model invoke with user prompt
    result = llm.invoke(user_input)
    # show result
    st.write(result.content)
