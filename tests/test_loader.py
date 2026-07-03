from tools.chunker import chunk_text
from tools.document_loader import load_document_text


def test_chunk_text_splits_with_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_chunk_text_prefers_word_boundaries() -> None:
    chunks = chunk_text("alpha beta gamma delta", chunk_size=12, overlap=3)
    assert chunks[0] == "alpha beta"
    assert chunks[1].startswith("gamma")


def test_load_document_text_reads_docx_without_extra_dependency(tmp_path) -> None:
    import zipfile

    docx_path = tmp_path / "demo.docx"
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Agentic AI module</w:t></w:r></w:p>
        <w:p><w:r><w:t>Planning and tool use</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/document.xml", xml)

    text = load_document_text(docx_path)

    assert "Agentic AI module" in text
    assert "Planning and tool use" in text


def test_load_document_text_reads_zip_but_skips_node_modules(tmp_path) -> None:
    import zipfile

    zip_path = tmp_path / "project.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("agentic-ai/README.md", "# Agentic AI\nUseful course notes.")
        archive.writestr("agentic-ai/node_modules/package/README.md", "Should be skipped.")

    text = load_document_text(zip_path)

    assert "Useful course notes" in text
    assert "Should be skipped" not in text
