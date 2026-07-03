import json

from tools.narrative_report import build_narrative_report, format_narrative_markdown
from tools.report_generator import format_report_markdown, save_report


def test_consistency_paper_gets_polished_narrative_sections() -> None:
    report = {
        "source": "demo.pdf",
        "summary": (
            "This paper introduces Position-Weighted Consistency, MT-Consistency, "
            "and Confidence-Aware Response Generation for multi-turn follow-up interactions."
        ),
        "claims": [
            {"claim": "CARG improves response stability in multi-turn settings."},
        ],
        "skeptical_findings": [
            {"severity": "medium", "finding": "The paper needs clearer statistical validation."}
        ],
        "evidence": [],
        "citation_checks": [],
    }

    narrative = build_narrative_report(report)
    markdown = format_narrative_markdown(report)

    section_titles = [section["title"] for section in narrative["sections"]]
    assert "Core Idea" in section_titles
    assert "Technical Contributions" in section_titles
    assert "Skeptical Notes" in section_titles
    assert "Position-Weighted Consistency" in markdown
    assert "## Model and App Limitations" in markdown


def test_save_report_preserves_existing_narrative_review(tmp_path) -> None:
    report = {
        "source": "module.pdf",
        "summary": "Generic summary.",
        "claims": [],
        "skeptical_findings": [],
        "evidence": [],
        "citation_checks": [],
        "narrative_review": {
            "opening": "Custom prepared opening.",
            "sections": [{"title": "Custom Section", "body": "Custom body."}],
        },
    }

    output_path = save_report(report, tmp_path, "module")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    markdown = format_report_markdown(report)

    assert saved["narrative_review"]["opening"] == "Custom prepared opening."
    assert "Custom Section" in markdown


def test_markdown_report_includes_model_limitations() -> None:
    report = {
        "source": "paper.pdf",
        "summary": "Generic summary.",
        "claims": [],
        "skeptical_findings": [],
        "evidence": [],
        "citation_checks": [],
    }

    markdown = format_report_markdown(report)

    assert "## Model and App Limitations" in markdown
    assert "Similarity search can find related text" in markdown
    assert "External verification is background context only" in markdown
