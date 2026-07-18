from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="conversational",
    max_new_tokens=100

)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Argentina")

print(result.content)
