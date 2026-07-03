from agents.evidence_finder import EvidenceFinder


def test_best_snippet_prefers_claim_sentence_over_header() -> None:
    finder = object.__new__(EvidenceFinder)
    text = (
        "Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions "
        "Anonymous Author(s) Affiliation Address email Abstract Large Language Models require consistency. "
        "Third, we introduce Confidence-Aware Response Generation (CARG), a framework that significantly "
        "improves response stability by integrating confidence scores."
    )
    claim = "Confidence-Aware Response Generation significantly improves response stability."

    snippet = finder._best_snippet(text, claim)

    assert "Confidence-Aware Response Generation" in snippet
    assert not snippet.startswith("Firm or Fickle")
