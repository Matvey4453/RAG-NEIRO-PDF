import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:3.8b-mini-4k-instruct-q6_k"

def ask(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    print("🤖 Тест Ollama")
    while True:
        q = input("\nВопрос: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\nОтвет:")
        print(ask(q))
