from tools.text_cleaner import clean_extracted_text, clean_display_text, clean_snippet_text


def test_clean_extracted_text_removes_line_numbers_and_joins_words() -> None:
    text = "Large Language Models show consis-\n4\ntency.\n15\nNext line."
    assert clean_extracted_text(text) == "Large Language Models show consistency.\nNext line."


def test_clean_display_text_repairs_common_mojibake() -> None:
    assert clean_display_text("modelâ€™s accuracy") == "model's accuracy"
    assert clean_display_text("responsesâ€”a characteristic") == "responses-a characteristic"


def test_clean_display_text_removes_inline_pdf_line_numbers() -> None:
    text = (
        "Large Language Models have shown remarkable capabilities across var- 1 ious tasks. "
        "Third, we introduce Confidence-Aware 10 Response Generation. "
        "The correct 111 3 answer should stay stable. "
        "214 As shown in Table 2, GPT demonstrates superior performance."
    )
    cleaned = clean_display_text(text)
    assert "various tasks" in cleaned
    assert "Confidence-Aware Response Generation" in cleaned
    assert "correct answer" in cleaned
    assert cleaned.endswith("As shown in Table 2, GPT demonstrates superior performance.")


def test_clean_display_text_restores_common_pdf_compounds() -> None:
    assert clean_display_text("earlystage stability in multiturn tasks") == "early-stage stability in multi-turn tasks"


def test_clean_snippet_text_trims_at_word_boundary() -> None:
    snippet = clean_snippet_text("one two three four", limit=11)
    assert snippet == "one two..."


def test_clean_snippet_text_marks_midword_starts() -> None:
    assert clean_snippet_text("onsulting demands consistency").startswith("...")


def test_clean_display_text_normalizes_mean_symbol_mojibake() -> None:
    assert clean_display_text("215 Â¯Rsway = 6.84") == "mean Rsway = 6.84"
