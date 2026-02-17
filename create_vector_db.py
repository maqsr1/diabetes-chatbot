import chromadb
from openai import OpenAI

client = OpenAI(api_key="sk-proj-XPdJp42oV-LDtKwIShXgFcmsj0gzziniY75oGcQvSLWO6QuAPixwfr1bbYsfZ9PQ8CXYZKkL_dT3BlbkFJw8pQuFbaypd0GuSoaqZfCRXl4xicLOxOArY3C2_QpN04_OskT8s2b0LwkCf6UJ1_ZQqGSH6GYAEY")

with open("documents.txt", "r") as f:
    texts = f.read().split("\n\n")

chroma_client = chromadb.PersistentClient(path="vector_db")

collection = chroma_client.get_or_create_collection(
    name="diabetes_docs"
)

def embed_text(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]

embeddings = embed_text(texts)

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(texts))]
)

print("Vector DB created successfully.")
