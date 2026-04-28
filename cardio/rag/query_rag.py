import ollama
import os
from cardio_llm import retrieve_cardio_context

OLLAMA_MODEL = os.getenv("CARDIO_OLLAMA_MODEL", "llama3.2:latest")

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
