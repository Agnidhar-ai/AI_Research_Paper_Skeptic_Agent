from pathlib import Path

from tools.text_cleaner import clean_extracted_text


def load_pdf_text(path: str | Path) -> str:
    pdf_path = Path(path)

    for loader in (_load_with_pymupdf, _load_with_pdfplumber, _load_with_pypdf):
        try:
            text = loader(pdf_path)
        except ModuleNotFoundError:
            continue
        if text.strip():
            return clean_extracted_text(text)

    raise RuntimeError(
        "Install PDF dependencies with `pip install -r requirements.txt` to load PDFs."
    )


def _load_with_pymupdf(path: Path) -> str:
    import fitz

    with fitz.open(path) as document:
        return "\n\n".join(page.get_text("text") for page in document)


def _load_with_pdfplumber(path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def _load_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
