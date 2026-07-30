from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()


model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

# step 1 classify the review

classifier_prompt = PromptTemplate(
    template="""
    You are a movie review classifier
    Classify the following review as either:
    -positive
    -negative
    -neutral
    -default
    Return only one word positive or negative
    Review: {review}
    """,
    input_variables=["review"]
)

classifier_chain = classifier_prompt | model | parser

# step 2


positive_prompt = PromptTemplate(
    template="""Reply to this positive movie review in a friendly way: \n (review)
    Review
    {review}
    """,
    input_variables=["review"]

)
negative_prompt = PromptTemplate(
    template="""
    reply to this negative movie review by aplogizing and offering relp: \n (review)
    review :
    {review}
    """,
    input_variables=["review"]
)
neutral_prompt = PromptTemplate(
    template="""
   Reply politely to this neutral review ask for suggestions to imporve
    Review :
    {review}
    """,
    input_variables=["review"]
)
default_prompt = PromptTemplate(
    template="""
        The sentiment could not be detemined.
        Review:
        {review}
    """,
    input_variables=["review"]
)

positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser
neutral_chain = neutral_prompt | model | parser
default_chain = default_prompt | model | parser

# review = "The movie was abosolutely Fantastic. I loved every minute of it."
# review = "The movie was 50/50"
review = "The movie was awful"

# step 3
sentiment = classifier_chain.invoke({"review": review})
print(sentiment)

conditional_chain = RunnableBranch(
    (lambda x: x["sentiment"].strip().lower() == "positive", positive_chain),
    (lambda x: x["sentiment"].strip().lower() == "neutral", neutral_chain),
    (lambda x: x["sentiment"].strip().lower() == "negative", negative_chain),
    default_chain
)

result = conditional_chain.invoke({
    "review": review,
    "sentiment": sentiment
})

print(result)
