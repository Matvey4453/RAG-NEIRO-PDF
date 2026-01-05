import faiss
import pickle
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:latest"  # phi3 ломается на формулах, gemma3 стабильнее

TOP_K = 3
MAX_CONTEXT_CHARS = 6000  # короче контекст = быстрее ответ
NUM_PREDICT = 300
TEMPERATURE = 0.3
STREAM = True
DEBUG = False  # True = показывать контекст перед запросом

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
def _looks_noisy(text: str) -> bool:
    # В учебных PDF формулы часто содержат символы вроде 〈 〉 μ и т.п.
    # Для phi3 на длинном контексте это иногда приводит к «каше».
    noisy_markers = "〈〉μ√∫≈≤≥"  # минимальный набор
    return any(ch in text for ch in noisy_markers)


def search(query: str, k: int = 3):
    q_emb = embedder.encode([query])
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)

    out = []  # list[tuple[score, idx, text]]
    for score, idx in zip(distances[0], indices[0]):
        if int(idx) == -1:
            continue
        out.append((float(score), int(idx), chunks[int(idx)]))
    return out

# ===== Запрос к Ollama =====
def ask_llama(prompt: str) -> str:
    """Предпочитаем /api/chat (корректный chat-template для instruct-моделей),
    иначе fallback на /api/generate."""

    options = {
        "num_predict": NUM_PREDICT,
        "temperature": TEMPERATURE,
    }

    # --- 1) Chat API ---
    chat_payload = {
        "model": MODEL_NAME,
        "stream": STREAM,
        "options": options,
        "messages": [
            {"role": "system", "content": "Ты — помощник. Отвечай строго по данному тексту. Если ответа нет — скажи: 'В документе нет информации'."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        if STREAM:
            with requests.post(OLLAMA_CHAT_URL, json=chat_payload, timeout=(10, 600), stream=True) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                full = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        import json

                        data = json.loads(line)
                    except Exception:
                        continue

                    msg = (data.get("message") or {}).get("content")
                    if msg:
                        print(msg, end="", flush=True)
                        full.append(msg)

                    if data.get("done") is True:
                        break
                print()
                return "".join(full).strip()
        else:
            r = requests.post(OLLAMA_CHAT_URL, json=chat_payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            return ((data.get("message") or {}).get("content") or "").strip()

    except requests.exceptions.HTTPError as e:
        # fallback на generate, если chat endpoint отсутствует
        if e.response is not None and e.response.status_code != 404:
            raise
    except requests.exceptions.ConnectionError:
        raise

    # --- 2) Fallback: Generate API ---
    gen_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": STREAM,
        "options": options,
    }

    try:
        if STREAM:
            with requests.post(OLLAMA_URL, json=gen_payload, timeout=(10, 600), stream=True) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                full = []
                import json

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    token = data.get("response")
                    if token:
                        print(token, end="", flush=True)
                        full.append(token)

                    if data.get("done") is True:
                        break
                print()
                return "".join(full).strip()
        else:
            r = requests.post(OLLAMA_URL, json=gen_payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "❌ Таймаут при генерации. Попробуйте уменьшить TOP_K/контекст или NUM_PREDICT."
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise RuntimeError(
                f"❌ Ошибка 404: endpoint '{OLLAMA_URL}' не найден или модель '{MODEL_NAME}' не доступна.\n"
                f"Проверьте: ollama serve, ollama list.\n"
                f"Детали: {e.response.text}"
            )
        raise

# ===== RAG =====
def answer(question):
    # Берём больше кандидатов и выкидываем «шумные» (формулы/спецсимволы)
    candidates = search(question, k=max(10, TOP_K))
    picked = []
    for _score, _idx, text in candidates:
        if _looks_noisy(text):
            continue
        picked.append(text)
        if len(picked) >= TOP_K:
            break

    # если всё отфильтровалось — берём как есть
    if not picked:
        picked = [t for _s, _i, t in candidates[:TOP_K]]

    context = "\n\n".join(picked)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[контекст обрезан]"

    if DEBUG:
        print(f"\n[DEBUG] Найдено чанков: {len(candidates)}, выбрано: {len(picked)}")
        print(f"[DEBUG] Длина контекста: {len(context)} символов")
        print(f"[DEBUG] Контекст:\n{context[:500]}...\n")

    prompt = (
        "ТЕКСТ:\n"
        f"{context}\n\n"
        "ВОПРОС:\n"
        f"{question}\n\n"
        "ТРЕБОВАНИЯ К ОТВЕТУ:\n"
        "- Ответь на русском языке.\n"
        "- Дай краткий ответ (2-4 предложения).\n"
        "- Если в тексте нет ответа, напиши ТОЛЬКО: 'Не нашёл ответ в документе. Задайте другой вопрос.'\n"
    )

    response = ask_llama(prompt)
    
    # Проверяем, не выдала ли модель "кашу" (много одиночных букв)
    import re
    words = re.findall(r'\S+', response)
    if len(words) > 20:
        single_letter_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        if single_letter_ratio > 0.3:
            return "❌ Модель выдала некорректный ответ. Попробуйте переформулировать вопрос или уточнить его."
    
    return response

# ===== Чат =====
def main():
    print("=" * 60)
    print("📚 RAG-система готова к работе!")
    print(f"🤖 Модель: {MODEL_NAME}")
    print(f"📄 Чанков в индексе: {len(chunks)}")
    print("💡 Команды: 'exit' или 'quit' для выхода")
    print("=" * 60)
    
    while True:
        q = input("\n❓ Вопрос: ")
        if q.lower() in ["exit", "quit"]:
            break

        print("\n💬 Ответ:")
        result = answer(q)
        if not STREAM:
            print(result)


if __name__ == "__main__":
    main()
