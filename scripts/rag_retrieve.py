import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = "kb"

def load_kb():
    docs = []
    names = []
    for fn in os.listdir(KB_DIR):
        if fn.lower().endswith(".txt"):
            path = os.path.join(KB_DIR, fn)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
            names.append(fn)
    return names, docs

def retrieve(query, top_k=2):
    names, docs = load_kb()
    if not docs:
        return []

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(docs)
    q = vec.transform([query])

    sims = cosine_similarity(q, X)[0]
    idxs = sims.argsort()[::-1][:top_k]

    results = []
    for i in idxs:
        results.append((names[i], float(sims[i]), docs[i]))
    return results

if __name__ == "__main__":
    q = input("Enter query: ").strip()
    hits = retrieve(q, top_k=2)
    print("\nTop matches:")
    for name, score, text in hits:
        print(f"\n--- {name} (score={score:.3f}) ---\n{text[:600]}")
