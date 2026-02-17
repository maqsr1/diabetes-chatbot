import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/chat")
def chat(question: Question):

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question.query
    )

    query_vector = np.array(
        [embedding.data[0].embedding]
    ).astype("float32")

    D, I = index.search(query_vector, k=3)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical data assistant."},
            {"role": "user", "content": question.query}
        ]
    )

    return {"answer": response.choices[0].message.content}