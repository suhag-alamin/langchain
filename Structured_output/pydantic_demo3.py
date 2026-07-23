from pydantic import BaseModel, EmailStr, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


# schema
class JobApplication(BaseModel):
    name: str = "Unknown"
    experience: Optional[int] = None
    email: EmailStr
    expected_salary: int = Field(
        gt=0, description="Expected annual salary of the candidate")


# model
model = ChatGroq(model="llama-3.3-70b-versatile")


# structured model
structured_model = model.with_structured_output(JobApplication)


# invoke
result = structured_model.invoke(
    """My name is Suhag. I have 3 years of experience
    as a Machine Learning Engineer.

    My email is arif@gmail.com.
    I am expecting an annual salary of 50000 dollars.
    """
)

print(result)
print(result.name)
