from config import settings


class LLMClient:
    def complete(self, prompt: str) -> str:
        provider = settings.llm_provider.lower()
        if provider == "openai":
            return self._complete_openai(prompt)
        if provider == "ollama":
            return self._complete_ollama(prompt)
        return ""

    def _complete_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _complete_ollama(self, prompt: str) -> str:
        import ollama

        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
