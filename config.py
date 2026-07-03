import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Research Paper Skeptic Agent"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    llm_provider: str = os.getenv("LLM_PROVIDER", "heuristic")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "memory")
    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "data/vector_db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    data_dir: str = os.getenv("DATA_DIR", "data")
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")


settings = Settings()
