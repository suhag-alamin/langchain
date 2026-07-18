from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "I love AI",
    "I love Machine Learning",
    "I love deep learning"
]


vector = embedding.embed_documents(documents)

print(len(vector))
print(vector[0])
