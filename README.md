# AI Research Paper Skeptic Agent

An experimental agent for reading research papers, surfacing weak claims, checking citations, finding supporting evidence, and producing a skeptical structured report.

## Quick Start

Requirements:

- Python 3.11+
- VS Code or another editor
- Git and GitHub for version control

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add PDFs, DOCX files, text/Markdown files, HTML files, or small ZIP archives to `data/papers`, then run:

```powershell
python app.py
```

Reports are written to `outputs/reports`.

To use the Streamlit UI:

```powershell
.\run_app.ps1
```

If PowerShell blocks scripts, run this instead:

```bat
run_app.bat
```

Both launchers use `.venv\Scripts\python.exe -m streamlit`, so they work even when the `streamlit`
command is not recognized globally. Keep the launcher window open, then open `http://localhost:8502`.

## Stack

- Python 3.11+
- Virtual environment with `venv`
- LangChain-compatible orchestration hooks
- `memory`, `faiss`, or `chroma` vector store backend
- PyMuPDF, pdfplumber, then pypdf PDF parsing fallback
- Built-in DOCX, text/Markdown/HTML, and safe ZIP text extraction
- OpenAI-compatible or Ollama-compatible LLM client
- Streamlit UI and CLI entry point
- Optional Wikipedia and arXiv external source checks
- Git + GitHub workflow

## Configuration

Set these values in `.env`:

```text
LLM_PROVIDER=heuristic
VECTOR_STORE_BACKEND=memory
```

Use `LLM_PROVIDER=openai` with `OPENAI_API_KEY` for OpenAI-compatible models, or `LLM_PROVIDER=ollama` with `OLLAMA_BASE_URL` for a local Ollama server.

Use `VECTOR_STORE_BACKEND=faiss` or `VECTOR_STORE_BACKEND=chroma` when you want LangChain-backed vector search.

The Streamlit app can optionally check Wikipedia and arXiv for background context. These checks are treated as supporting context, not proof of a paper's claims.
External checks now show whether background sources were found, only partially checked, unavailable, or returned no usable results.

The paper chat keeps a lightweight discussion memory, shows answer sources and confidence, and only uses web search when the user enables the web-search checkbox.

The Streamlit interface now includes an interactive research workspace with a paper map, section-focused study view, reading modes, understanding levels, progress tracking, notes, source confidence, and a lightweight knowledge graph.

## Known Limitations

- The current default is heuristic. It is explainable and cheap, but less flexible than a strong LLM.
- Similarity search can find related text, but related text is not always direct proof.
- PDF extraction can be messy when papers contain tables, formulas, columns, or broken encoding.
- Citation checks detect citation markers, but they do not read every referenced paper.
- External verification is background context only. It should not be treated as validation of the paper experiment.
- Use the saved-report selector carefully so you restore the intended paper, not just the newest report.

## Project Layout

- `agents/`: agent orchestration and review logic
- `prompts/`: system and task prompts
- `tools/`: PDF loading, chunking, retrieval, embeddings, vector store, report output
- `rag/`: ingestion and querying helpers
- `memory/`: simple conversation state
- `tests/`: starter tests

## Current Workflow

Documents move through loading, text chunking, a memory, FAISS, or Chroma knowledge base, claim/topic analysis, evidence retrieval, reasoning checks, and JSON report generation.
