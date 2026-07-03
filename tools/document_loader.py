from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from tools.pdf_loader import load_pdf_text
from tools.text_cleaner import clean_extracted_text


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".zip",
}

SKIPPED_ZIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
}

READABLE_ZIP_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".html",
    ".htm",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}


def load_document_text(path: str | Path) -> str:
    document_path = Path(path)
    suffix = document_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf_text(document_path)
    if suffix == ".docx":
        return _load_docx_text(document_path)
    if suffix in {".txt", ".md", ".markdown"}:
        return clean_extracted_text(document_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix in {".html", ".htm"}:
        return clean_extracted_text(_strip_html(document_path.read_text(encoding="utf-8", errors="ignore")))
    if suffix == ".zip":
        return _load_zip_text(document_path)

    supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
    raise ValueError(f"Unsupported document type '{suffix}'. Supported types: {supported}")


def _load_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_text = archive.read("word/document.xml")
    except KeyError as exc:
        raise RuntimeError("DOCX file does not contain word/document.xml.") from exc

    root = ET.fromstring(xml_text)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        pieces = [
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
        ]
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return clean_extracted_text("\n\n".join(paragraphs))


def _load_zip_text(path: Path, max_files: int = 40, max_chars: int = 60000) -> str:
    collected = []
    total_chars = 0
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir() or _skip_zip_member(item.filename):
                continue
            suffix = Path(item.filename).suffix.lower()
            if suffix not in READABLE_ZIP_EXTENSIONS:
                continue
            try:
                raw = archive.read(item, pwd=None)
            except RuntimeError:
                continue
            text = raw.decode("utf-8", errors="ignore")
            if suffix in {".html", ".htm"}:
                text = _strip_html(text)
            text = clean_extracted_text(text)
            if not text:
                continue
            snippet = text[:8000]
            collected.append(f"File: {item.filename}\n{snippet}")
            total_chars += len(snippet)
            if len(collected) >= max_files or total_chars >= max_chars:
                break

    if not collected:
        return "No readable text files were found in this ZIP archive."
    return clean_extracted_text("\n\n".join(collected))


def _skip_zip_member(name: str) -> bool:
    parts = {part.lower() for part in Path(name).parts}
    return bool(parts.intersection(SKIPPED_ZIP_PARTS))


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)
