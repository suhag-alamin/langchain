from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()


model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()


positive_prompt = ChatPromptTemplate.from_template(
    "Reply to this positive movie review in a friendly way: \n {review}")
negative_prompt = ChatPromptTemplate.from_template(
    "Reply to this negative movie review by apologizing and offering help: \n {review}")

positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser

conditional_chain = RunnableBranch(
    (lambda x: "good" in x["review"].lower(), positive_chain),
    negative_chain
)

result = conditional_chain.invoke({
    "review": "Movie was good"
})

print(result)
