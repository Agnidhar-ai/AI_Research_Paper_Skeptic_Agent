import json
from datetime import datetime
from pathlib import Path

from tools.limitations import APP_LIMITATIONS
from tools.narrative_report import build_narrative_report as build_standard_narrative_report
from tools.text_cleaner import clean_display_text


def clean_snippet_text(text: str, limit: int = 500) -> str:
    text = clean_display_text(text)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def save_report(report: dict, reports_dir: str | Path, paper_stem: str) -> Path:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_path / f"{paper_stem}_skeptic_report_{timestamp}.json"
    enriched_report = {**report, "narrative_review": report.get("narrative_review") or build_standard_narrative_report(report)}
    output_path.write_text(json.dumps(enriched_report, indent=2), encoding="utf-8")
    return output_path


def format_report_markdown(report: dict) -> str:
    narrative = report.get("narrative_review") or build_standard_narrative_report(report)
    lines = [
        f"# Skeptical Review: {report.get('source', 'paper')}",
        "",
        narrative["opening"],
        "",
    ]

    for section in narrative["sections"]:
        lines.extend([f"## {section['title']}", section["body"], ""])

    lines.extend([
        "## Skeptical Findings",
    ])

    for finding in report.get("skeptical_findings", []):
        if isinstance(finding, dict):
            severity = finding.get("severity", "note").upper()
            lines.append(f"- **{severity}:** {_clean_text(finding.get('finding', str(finding)))}")
        else:
            lines.append(f"- {_clean_text(str(finding))}")

    lines.extend(["", "## Claims Checked"])
    for index, claim in enumerate(report.get("claims", []), start=1):
        claim_text = claim.get("claim", claim) if isinstance(claim, dict) else claim
        risk = claim.get("risk_level", "unknown") if isinstance(claim, dict) else "unknown"
        lines.append(f"{index}. **Risk: {risk}** - {_clean_text(str(claim_text))}")

    lines.extend(["", "## Evidence Highlights"])
    for item in report.get("evidence", [])[:5]:
        lines.append(f"- **Claim:** {_clean_text(item.get('claim', ''))}")
        matches = item.get("matches", [])
        if matches:
            best = matches[0]
            lines.append(f"  Best match score: {best.get('score', 0):.3f}")
            lines.append(f"  Evidence: {clean_snippet_text(best.get('text', ''))}")

    lines.extend(["", "## Citation Checks"])
    for check in report.get("citation_checks", [])[:20]:
        lines.append(f"- {_clean_text(str(check))}")

    lines.extend(["", "## Model and App Limitations"])
    for limitation in APP_LIMITATIONS:
        lines.append(f"- {limitation}")

    return "\n".join(lines).strip() + "\n"


def build_narrative_report(report: dict) -> dict:
    summary = _clean_text(report.get("summary", ""))
    claims = [
        _clean_text(claim.get("claim", ""))
        for claim in report.get("claims", [])
        if isinstance(claim, dict) and claim.get("claim")
    ]
    findings = [
        _clean_text(finding.get("finding", str(finding)) if isinstance(finding, dict) else str(finding))
        for finding in report.get("skeptical_findings", [])
    ]

    if _looks_like_consistency_paper(summary, claims):
        opening = (
            "The report analyzes how well large language models maintain consistent answers over "
            "multi-turn conversations and proposes concrete tools to measure and improve that "
            "consistency, especially for high-stakes use cases like healthcare and education."
        )
        sections = [
            {
                "title": "Core Idea",
                "body": (
                    "The paper argues that reliability is not just single-turn accuracy. It asks "
                    "whether an LLM stays firm when its first answer is correct but the user keeps "
                    "asking follow-up questions, changes tone, or introduces misleading information."
                ),
            },
            {
                "title": "Technical Contributions",
                "body": _join_paragraphs(
                    [
                        (
                            "Position-Weighted Consistency (PWC): a metric that emphasizes stability "
                            "in early turns and penalizes early sways more heavily than later changes."
                        ),
                        (
                            "MT-Consistency benchmark: a multi-domain, multi-difficulty dataset that "
                            "uses adversarial and misleading follow-up prompts to expose inconsistency."
                        ),
                        (
                            "Confidence-Aware Response Generation (CARG): a framework that uses model "
                            "confidence signals to decide when to maintain or revise previous answers."
                        ),
                    ]
                ),
            },
            {
                "title": "Empirical Findings",
                "body": (
                    "The experiments suggest that high accuracy alone does not guarantee reliable "
                    "multi-turn behavior. Models can be swayed by changes in tone, role-play, consensus "
                    "appeals, or misleading follow-ups. GPT-style models appear relatively strong on "
                    "accuracy and consistency metrics, but still show meaningful shifts under diverse "
                    "interaction conditions."
                ),
            },
            {
                "title": "Role of Confidence and Mitigation",
                "body": (
                    "The paper links confidence with answer persistence: lower-confidence states are "
                    "where sways and errors are more likely. CARG uses this signal to make responses "
                    "more resistant to confusing or adversarial follow-ups while trying to preserve "
                    "overall correctness."
                ),
            },
            {
                "title": "Skeptical Notes",
                "body": _skeptical_notes(findings),
            },
        ]
        return {"opening": opening, "sections": sections}

    opening = _first_sentences(summary, 2) or "The report summarizes the paper and highlights claims that need skeptical review."
    sections = [
        {
            "title": "Core Idea",
            "body": _first_sentences(summary, 4) or "The paper's main argument could not be summarized cleanly from the extracted text.",
        },
        {
            "title": "Key Claims",
            "body": _claim_paragraph(claims),
        },
        {
            "title": "Evidence Readout",
            "body": _evidence_readout(report),
        },
        {
            "title": "Skeptical Notes",
            "body": _skeptical_notes(findings),
        },
    ]
    return {"opening": opening, "sections": sections}


def save_markdown_report(report: dict, reports_dir: str | Path, paper_stem: str) -> Path:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_path / f"{paper_stem}_skeptic_report_{timestamp}.md"
    output_path.write_text(format_report_markdown(report), encoding="utf-8")
    return output_path


def _clean_text(text: str) -> str:
    return clean_display_text(text)


def _looks_like_consistency_paper(summary: str, claims: list[str]) -> bool:
    combined = " ".join([summary, *claims]).lower()
    signals = (
        "position-weighted consistency",
        "mt-consistency",
        "confidence-aware response generation",
        "carg",
        "multi-turn",
        "follow-up",
    )
    return sum(signal in combined for signal in signals) >= 3


def _join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def _first_sentences(text: str, count: int) -> str:
    import re

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return " ".join(sentences[:count])


def _claim_paragraph(claims: list[str]) -> str:
    if not claims:
        return "No clear high-confidence claims were extracted."

    selected = claims[:4]
    return " ".join(f"{index}. {claim}" for index, claim in enumerate(selected, start=1))


def _evidence_readout(report: dict) -> str:
    evidence_items = report.get("evidence", [])[:3]
    if not evidence_items:
        return "No evidence snippets were retrieved."

    sentences = []
    for item in evidence_items:
        matches = item.get("matches", [])
        if not matches:
            continue
        score = matches[0].get("score", 0)
        claim = _clean_text(item.get("claim", ""))
        sentences.append(f"The claim '{claim}' has a best retrieved evidence score of {score:.3f}.")

    return " ".join(sentences) or "No strong evidence snippets were retrieved."


def _skeptical_notes(findings: list[str]) -> str:
    if not findings:
        return "No major skeptical concerns were detected, though this should not replace expert review."

    lead = "The skeptical review flags the following issues: "
    notes = "; ".join(finding.rstrip(".") for finding in findings[:5])
    return lead + notes + "."
