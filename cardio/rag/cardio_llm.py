from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import os

DB_PATH = "vectorstore"
OLLAMA_MODEL = os.getenv("CARDIO_OLLAMA_MODEL", "llama3.2:latest")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def retrieve_cardio_context(query):
    results = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def generate_cardio_answer(query):
    context = retrieve_cardio_context(query)

    prompt = f"""
You are a cardiology AI assistant.

Use only the provided context.
Do not give final medical diagnosis.
Always say that a doctor/cardiologist review is required.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
