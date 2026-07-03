from agents.claim_analyzer import ClaimAnalyzer


def test_claim_analyzer_extracts_signal_claims() -> None:
    claims = ClaimAnalyzer().extract_claims(
        "The introduction is short. Our method demonstrates significant improvement over strong baselines."
    )
    assert claims[0]["risk_level"] == "medium"
    assert "significant" in claims[0]["signals"]


def test_claim_analyzer_extracts_topics_when_slide_deck_has_no_claims() -> None:
    claims = ClaimAnalyzer().extract_claims(
        "LangChain EcoSystem LangChain Core LangChain Community LangSmith LangServe Model Abstraction Layer"
    )

    assert claims[0]["signals"] == ["topic"]
    assert "LangChain" in claims[0]["claim"]
