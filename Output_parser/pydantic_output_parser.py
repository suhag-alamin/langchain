from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# schema


class ModelEvaluation(BaseModel):
    model_name: str = Field(description="Name of the machine learning model")
    accuracy: float = Field(gt=0, lt=1, description="Accuracy of the model")
    dataset: str = Field(description="Name of the dataset used for evaluation")


# parser

parser = PydanticOutputParser(
    pydantic_object=ModelEvaluation
)

# prompt template

template = PromptTemplate(
    template="""
    Generate the name, accuracy and dataset of a fictional machine learning model trained for {task}
    {format_instruction}
    """,
    input_variables=["task"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }


)


# chain
chain = template | model | parser

result = chain.invoke({
    "task": "image_classification"
})

print(result)
print(result.model_name)
