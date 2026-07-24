from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="""
    Give me 5 facts about {topic}.
    {format_instruction}
    """,
    input_variables=[
        {
            "topic"
        }
    ],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


# prompt = template.format(topic="Machine learning")

# print(prompt)

chain = template | model | parser

result = chain.invoke({"topic": "Machine learning"})

print(result)
print(type(result))
