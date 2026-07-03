# Architecture

The app is organized around a simple pipeline:

1. Load PDF text from `data/papers`.
2. Split text into chunks for retrieval.
3. Store chunks in a memory, FAISS, or Chroma knowledge base.
4. Extract likely claims from the paper.
5. Retrieve supporting context for each claim.
6. Run reasoning checks for assumptions, contradictions, weak evidence, missing citations, and over-strong conclusions.
7. Save structured reports to `outputs/reports`.

The default implementation uses local heuristic review logic and lightweight TF-IDF retrieval so it can run without external services. Setting `VECTOR_STORE_BACKEND=faiss` or `VECTOR_STORE_BACKEND=chroma` switches retrieval to a LangChain-compatible vector store path.

LLM use is optional. Set `LLM_PROVIDER=openai` for OpenAI-compatible APIs or `LLM_PROVIDER=ollama` for a local Ollama server. The heuristic path remains the default so the project works before external credentials or local models are configured.

## Known Limitations

- The current default is heuristic. It is explainable and cheap, but less flexible than a strong LLM.
- Similarity search can find related text, but related text is not always direct proof.
- PDF extraction can be messy when papers contain tables, formulas, columns, or broken encoding.
- Citation checks detect citation markers, but they do not read every referenced paper.
- External verification is background context only. It should not be treated as validation of the paper experiment.
- The app uses a saved-report selector so users can deliberately restore the intended paper instead of accidentally reopening only the newest report.
