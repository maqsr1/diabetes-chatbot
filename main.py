from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb.config import Settings

app = FastAPI()

client = OpenAI(api_key="sk-proj-XPdJp42oV-LDtKwIShXgFcmsj0gzziniY75oGcQvSLWO6QuAPixwfr1bbYsfZ9PQ8CXYZKkL_dT3BlbkFJw8pQuFbaypd0GuSoaqZfCRXl4xicLOxOArY3C2_QpN04_OskT8s2b0LwkCf6UJ1_ZQqGSH6GYAY")

# Load ChromaDB
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="vector_db"
    )
)

collection = chroma_client.get_or_create_collection("diabetes_docs")

class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Diabetes Chatbot API is running"}

@app.post("/chat")
def chat(question: Question):
    # Embed user query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question.query
    ).data[0].embedding

    # Query vector DB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question.query}"}
        ]
    )

    return {"answer": response.choices[0].message.content}
