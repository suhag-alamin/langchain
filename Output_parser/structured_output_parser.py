from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser, ResponseSchema)
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# schema

response_schemas = [
    ResponseSchema(
        name="Fact_1",
        description="The first fact about the topic"
    ),
    ResponseSchema(
        name="Fact_2",
        description="The second fact about the topic"
    ),
    ResponseSchema(
        name="Fact_3",
        description="The third fact about the topic"
    ),
    ResponseSchema(
        name="Fact_4",
        description="The forth fact about the topic"
    ),
    ResponseSchema(
        name="Fact_5",
        description="The fifth fact about the topic"
    )
]

# parser

parser = StructuredOutputParser.from_response_schemas(
    response_schemas)


# prompt

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

# chain

chain = template | model | parser

# invokke

result = chain.invoke({
    'topic': "AI"
})

print(result['Fact_5'])
