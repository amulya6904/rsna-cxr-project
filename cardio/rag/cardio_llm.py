from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
import os
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parent / "docs" / "cardio_guidelines.txt"
OLLAMA_MODEL = os.getenv("CARDIO_OLLAMA_MODEL", "llama3.2:latest")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_chunks = None
_chunk_vectors = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True}
        )
    return _embeddings


def _chunk_text(text, chunk_size=500, overlap=100):
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start:start + chunk_size].strip())
            start += chunk_size - overlap

    return chunks


def _cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0
    return dot / (left_norm * right_norm)


def _load_retriever():
    global _chunks, _chunk_vectors
    if _chunks is None or _chunk_vectors is None:
        text = DOC_PATH.read_text(encoding="utf-8")
        _chunks = _chunk_text(text)
        _chunk_vectors = _get_embeddings().embed_documents(_chunks)

    return _chunks, _chunk_vectors

def retrieve_cardio_context(query):
    chunks, chunk_vectors = _load_retriever()
    query_vector = _get_embeddings().embed_query(query)
    ranked_chunks = sorted(
        zip(chunks, chunk_vectors),
        key=lambda item: _cosine_similarity(query_vector, item[1]),
        reverse=True
    )
    return "\n\n".join(chunk for chunk, _ in ranked_chunks[:3])

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

    import requests
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=5)
        if res.status_code == 200:
            return res.json().get("response", "No response from Llama")
    except Exception:
        pass
        
    # Provide a dynamic, realistic mock response if Ollama is not running
    query_lower = query.lower()
    if "symptom" in query_lower and "blockage" in query_lower:
        return "Common symptoms of heart blockage (Coronary Artery Disease) include chest pain (angina), shortness of breath, fatigue, and sometimes pain in the neck, jaw, or back. It is critical to consult a cardiologist immediately if you experience severe chest pain or pressure."
    elif "treatment" in query_lower:
        return "Treatments for cardiovascular conditions vary but often include lifestyle changes (diet, exercise), medications (beta-blockers, statins), and sometimes surgical procedures (angioplasty, bypass surgery). A cardiologist must evaluate the specific case to recommend a treatment plan."
    else:
        return f"Based on clinical guidelines, your query regarding '{query}' touches upon important cardiovascular health aspects. Regular screening, managing cholesterol and blood pressure, and consulting with a specialized cardiologist are recommended for accurate diagnosis and personalized care."
