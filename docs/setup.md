# Setup

## Required Tools

- Python 3.11+
- VS Code
- Git and a GitHub account

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the example environment file when you want local settings:

```powershell
Copy-Item .env.example .env
```

## LLM Options

Default local heuristic mode:

```text
LLM_PROVIDER=heuristic
```

OpenAI-compatible mode:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
CHAT_MODEL=gpt-4.1-mini
```

Ollama mode:

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
CHAT_MODEL=llama3.1
```

## Vector Store Options

```text
VECTOR_STORE_BACKEND=memory
VECTOR_STORE_BACKEND=faiss
VECTOR_STORE_BACKEND=chroma
```

## Tests

```powershell
python -m pytest --basetemp .pytest_tmp
```

GitHub Actions runs the same test command on pushes and pull requests to `main`.
