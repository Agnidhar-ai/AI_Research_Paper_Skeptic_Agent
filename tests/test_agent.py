from agents.skeptic_agent import SkepticAgent


def test_skeptic_agent_returns_report_shape() -> None:
    report = SkepticAgent().review(
        "This paper demonstrates a novel model. The dataset has limitations. "
        "Results show significant improvement over a baseline [1].",
        "demo.pdf",
    )
    assert report["source"] == "demo.pdf"
    assert "claims" in report
    assert "evidence" in report
    assert "reasoning" in report
    assert "skeptical_findings" in report
    assert "citation_checks" in report
