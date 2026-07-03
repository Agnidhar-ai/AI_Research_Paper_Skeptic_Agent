# Workflow

1. Put research paper PDFs in `data/papers`.
2. Run `python app.py`.
3. Review generated JSON reports in `outputs/reports`.
4. Each report includes extracted claims, retrieved evidence snippets, reasoning checks, skeptical findings, and citation checks.
5. Use RAG helpers in `rag/ingest.py` and `rag/query.py` for targeted paper questions.

For the UI, run:

```powershell
.\run_app.ps1
```

If PowerShell blocks scripts, use:

```bat
run_app.bat
```

Keep the launcher window open while using the app, then open `http://localhost:8502`.

If you want to reopen previous work, use the saved-report selector in the app and choose the exact report by paper name, timestamp, and filename. Do not assume the latest saved report is the one you meant to restore.

For version control:

```powershell
git init
git add .
git commit -m "Initial research skeptic agent scaffold"
```

Suggested next steps:

- Add citation extraction for author-year formats.
- Generate Markdown or PDF review reports.
