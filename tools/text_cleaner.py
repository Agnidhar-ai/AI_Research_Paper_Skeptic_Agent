import re


BAD_ENCODING_MARKERS = ("Ã", "Â", "â", "Î", "Ï")


def clean_extracted_text(text: str) -> str:
    text = repair_mojibake(text)
    text = _normalize_punctuation(text)
    text = _remove_standalone_line_numbers(text)
    text = _remove_inline_line_numbers(text)
    text = _join_hyphenated_line_breaks(text)
    text = _restore_common_compounds(text)
    text = _normalize_line_breaks(text)
    return text.strip()


def clean_display_text(text: str) -> str:
    text = repair_mojibake(text)
    text = _normalize_punctuation(text)
    text = _remove_inline_line_numbers(text)
    text = _restore_common_compounds(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_snippet_text(text: str, limit: int = 500) -> str:
    text = clean_display_text(text)
    first_word = text.split(" ", 1)[0] if text else ""
    if first_word and first_word[0].islower() and len(first_word) > 6:
        text = "..." + text
    text = re.sub(r"\bteaching\s+\d{1,4}\b", "teaching", text)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def repair_mojibake(text: str) -> str:
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    if _bad_marker_count(repaired) < _bad_marker_count(text):
        return repaired
    return text


def _remove_standalone_line_numbers(text: str) -> str:
    return re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", text)


def _remove_inline_line_numbers(text: str) -> str:
    text = re.sub(r"([A-Za-z])-\s+\d{1,4}\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<=[A-Za-z])\s+\d{1,4}(?:\s+\d{1,4})+\s+(?=[A-Za-z])", " ", text)
    text = re.sub(r"^\s*\d{1,4}\s+(?=mean\s+[A-Za-z])", "", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\s+(?=[A-Z][a-z])", "", text)
    text = re.sub(r"(?<=[a-z,;:.)])\s+\d{1,4}\s+(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[a-z,;:.)])\s+\d{1,4}\s+(?=mean\s+[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])\s+\d{1,4}\s+(?=[a-z])", " ", text)
    return text


def _join_hyphenated_line_breaks(text: str) -> str:
    return re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)


def _normalize_line_breaks(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _normalize_punctuation(text: str) -> str:
    replacements = {
        "Â¯": "mean ",
        "â€”": "-",
        "â€“": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u00af": "mean ",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "ï¬€": "ff",
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ï¬ƒ": "ffi",
        "ï¬„": "ffl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _restore_common_compounds(text: str) -> str:
    replacements = {
        "earlystage": "early-stage",
        "multiturn": "multi-turn",
        "followup": "follow-up",
        "highstakes": "high-stakes",
        "zeroshot": "zero-shot",
        "realworld": "real-world",
    }
    for compact, compound in replacements.items():
        text = re.sub(rf"\b{compact}\b", compound, text, flags=re.IGNORECASE)
    return text


def _bad_marker_count(text: str) -> int:
    return sum(text.count(marker) for marker in BAD_ENCODING_MARKERS)
