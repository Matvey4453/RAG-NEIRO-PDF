import faiss
import pickle
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b-instruct-q6_K"

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

# ===== Загружаем индекс =====
print("📦 Загружаем FAISS...")
index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ===== Поиск =====
def search(query, k=3):
    q_emb = embedder.encode([query])
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    out = []
    for i in indices[0]:
        if i == -1:
            continue
        out.append(chunks[i])
    return out

# ===== Запрос к Ollama =====
def ask_llama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response keys: {list(data.keys())}")
    return data["response"]

# ===== RAG =====
def answer(question):
    hits = search(question)
    context = "\n\n".join(hits)

    prompt = f"""
Ты — помощник.
Отвечай ТОЛЬКО на основе текста ниже.
Если ответа нет — скажи: "В документе нет информации".

ТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""

    return ask_llama(prompt)

# ===== Чат =====
while True:
    q = input("\n❓ Вопрос: ")
    if q.lower() in ["exit", "quit"]:
        break

    print("\n💬 Ответ:")
    print(answer(q))
