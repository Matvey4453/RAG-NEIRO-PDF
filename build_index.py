import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "doc.pdf"
INDEX_PATH = BASE_DIR / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

# ===== 1. Читаем PDF =====
def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ===== 2. Чанки =====
def split_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks

print("📄 Читаем PDF...")
text = read_pdf(str(PDF_PATH))

print("✂️ Разбиваем на чанки...")
chunks = split_text(text)
print(f"🧩 Чанков: {len(chunks)}")

# ===== 3. Эмбеддинги =====
print("🧠 Загружаем embedding модель...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

print("🔢 Создаём эмбеддинги...")
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.asarray(embeddings, dtype="float32")

# Для семантического поиска обычно лучше косинусная близость.
# Реализуем её через inner product по L2-нормализованным векторам.
faiss.normalize_L2(embeddings)

# ===== 4. FAISS =====
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ===== 5. Сохраняем =====
faiss.write_index(index, str(INDEX_PATH))

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)
    
print("✅ Индекс сохранён (index.faiss + chunks.pkl)")
