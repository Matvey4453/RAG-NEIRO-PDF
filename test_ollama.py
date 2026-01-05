import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:3.8b-mini-4k-instruct-q6_k"

print("🔍 Тестируем подключение к Ollama...")

payload = {
    "model": MODEL_NAME,
    "prompt": "Hello, respond with just 'Hi'",
    "stream": False
}

try:
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    print(f"Статус код: {r.status_code}")
    print(f"Ответ: {r.text[:500]}")
    r.raise_for_status()
    data = r.json()
    print(f"\n✅ Успех! Ответ модели: {data.get('response', 'Нет ответа')}")
except requests.exceptions.HTTPError as e:
    print(f"\n❌ HTTP ошибка: {e}")
    print(f"Детали ответа: {e.response.text}")
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
