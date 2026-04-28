from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import os

DB_PATH = "vectorstore"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

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
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def generate_answer(query):
    context = retrieve_cardio_context(query)

    prompt = f"""
You are a cardiology assistant.

Use the context below to answer the question.

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

if __name__ == "__main__":
    question = "What should a high-risk heart patient do?"
    answer = generate_answer(question)

    print("Answer:")
    print("-------")
    print(answer)
