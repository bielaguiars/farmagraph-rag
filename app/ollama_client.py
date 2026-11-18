import json

import requests


class OllamaClient:
    """Cliente mínimo da API HTTP do Ollama (LLM e embeddings locais)."""

    def __init__(self, base_url: str, chat_model: str, embed_model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Falha ao chamar o Ollama ({path}): {exc}") from exc

    def models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except requests.RequestException:
            return []

    # nomic-embed-text recomenda prefixos de tarefa distintos para documentos e consultas
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._post(
            "/api/embed",
            {"model": self.embed_model, "input": texts, "truncate": True, "keep_alive": "30m"},
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("O Ollama não retornou um embedding para cada texto.")
        return embeddings

    def generate(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        num_predict: int = 900,
        num_ctx: int = 8192,
    ) -> str:
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # saída estruturada (JSON Schema) do Ollama
        result = self._post("/api/generate", payload)
        text = result.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("O Ollama retornou uma resposta vazia.")
        return text.strip()

    def generate_json(self, prompt: str, json_schema: dict, **kwargs) -> dict:
        raw = self.generate(prompt, json_schema=json_schema, **kwargs)
        start = raw.find("{")
        if start < 0:
            raise RuntimeError("A resposta do LLM não contém JSON.")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON inválido retornado pelo LLM: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("O LLM precisa retornar um objeto JSON.")
        return parsed
