from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers",
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)",
     "Long (detailed explanation)"]
)


template = PromptTemplate.from_template(
    """ 
        Please summarize the research paper titled \"{paper_input}\" with the following specifications:\n        Explanation Style: {style_input}\n        Explanation Length: {length_input}\n        1. Mathematical Details:\n        - Include relevant mathematical equations if present in the paper.\n        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.\n        2. Analogies:\n        - Use relatable analogies to simplify complex ideas.\n        If certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.\n        Ensure the summary is clear, accurate, and aligned with the provided style and length.
        """

)


if st.button("Summarize"):
    # fill the placeholder
    prompt = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })

    # result
    result = model.invoke(prompt)
    st.write(result.content)
