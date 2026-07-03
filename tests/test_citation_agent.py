from agents.citation_agent import CitationAgent


def test_citation_agent_groups_citation_details() -> None:
    text = (
        "Prior work motivates this problem [1, 2, 3]. "
        "Results demonstrate significant improvement [4]."
    )
    details = CitationAgent().extract_citation_details(text)

    assert details[0]["citation"] == "[1, 2, 3]"
    assert details[0]["count"] == 3
    assert "Grouped citation" in details[0]["check"]
    assert "Claim-bearing" in details[1]["check"]
