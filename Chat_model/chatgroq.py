from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="openai/gpt-oss-120b",
)

# result = llm.invoke("What is the capital of Argentina")
result = llm.invoke("What is 3453425 + 646546?")

print(result.content)
