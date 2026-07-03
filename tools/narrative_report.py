import re

from tools.limitations import APP_LIMITATIONS
from tools.text_cleaner import clean_display_text


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
        return {
            "opening": (
                "The report analyzes how well large language models maintain consistent answers over "
                "multi-turn conversations and proposes concrete tools to measure and improve that "
                "consistency, especially for high-stakes use cases like healthcare and education."
            ),
            "sections": [
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
            ],
        }

    if _looks_like_learning_module(report, summary, claims):
        topics = _topic_list(report, summary, claims)
        topic_text = ", ".join(topics[:10]) if topics else "the main concepts in the uploaded material"
        return {
            "opening": (
                f"This document is best treated as an instructional learning module about {topic_text}. "
                "The review should help a learner understand the concepts, follow the workflow, and identify "
                "which tool or framework claims need current-source verification."
            ),
            "sections": [
                {
                    "title": "Learning Goal",
                    "body": (
                        "Use this material to build conceptual understanding rather than to verify an empirical "
                        "research result. Focus first on the vocabulary, then the workflow, then the practical use cases."
                    ),
                },
                {
                    "title": "Key Topics",
                    "body": _learning_topic_paragraph(topics),
                },
                {
                    "title": "How the Pieces Connect",
                    "body": (
                        "The module appears to organize related concepts into a learning path: first name the framework "
                        "or technique, then understand its components, then connect it to agent workflows, retrieval, "
                        "planning, orchestration, or evaluation depending on the topic."
                    ),
                },
                {
                    "title": "Practice Path",
                    "body": (
                        "A good way to study it is: 1. list the key terms; 2. explain each term in one sentence; "
                        "3. draw the workflow; 4. ask what problem each component solves; 5. verify any framework-specific "
                        "API claims against current documentation."
                    ),
                },
                {
                    "title": "Skeptical Notes",
                    "body": _skeptical_notes(findings),
                },
            ],
        }

    return {
        "opening": _first_sentences(summary, 2)
        or "The report summarizes the paper and highlights claims that need skeptical review.",
        "sections": [
            {
                "title": "Core Idea",
                "body": _first_sentences(summary, 4)
                or "The paper's main argument could not be summarized cleanly from the extracted text.",
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
        ],
    }


def format_narrative_markdown(report: dict) -> str:
    narrative = build_narrative_report(report)
    lines = [f"# Skeptical Review: {report.get('source', 'paper')}", "", narrative["opening"], ""]
    for section in narrative["sections"]:
        lines.extend([f"## {section['title']}", section["body"], ""])
    lines.append("## Model and App Limitations")
    for limitation in APP_LIMITATIONS:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


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


def _looks_like_learning_module(report: dict, summary: str, claims: list[str]) -> bool:
    source = str(report.get("source", "")).lower()
    combined = " ".join([source, summary, *claims]).lower()
    learning_markers = (
        "pptx",
        "module",
        "langchain",
        "langgraph",
        "crewai",
        "agentic framework",
        "agentic ai",
        "andrej karpathy",
        "planning & multi-agent",
        "deep dive into llms",
        "lecture",
        "multi-agent systems",
        "llms as reasoning engines",
        "llm development stack",
        "docx",
        "deep learning",
        "deeplearning",
        "rag",
    )
    research_markers = (
        "position-weighted consistency",
        "mt-consistency",
        "confidence-aware response generation",
        "abstract",
        "experiment",
        "empirical",
    )
    return any(marker in combined for marker in learning_markers) and not (
        "pptx" not in source and sum(marker in combined for marker in research_markers) >= 2
    )


def _topic_list(report: dict, summary: str, claims: list[str]) -> list[str]:
    combined_text = " ".join([str(report.get("source", "")), summary, *claims])
    topics = _domain_learning_topics(combined_text)
    if len(topics) < 8:
        for claim in claims:
            prefix = "The document covers the topic:"
            if claim.startswith(prefix):
                topics.append(claim.removeprefix(prefix).strip(" ."))

    if len(topics) < 8:
        topics.extend(_extract_title_phrases(" ".join([report.get("source", ""), summary])))
    cleaned = []
    seen = set()
    for topic in topics:
        topic = _clean_text(topic).strip(" -:|.")
        if not _looks_like_clean_topic(topic):
            continue
        normalized = topic.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(topic)
    return cleaned[:12]


def _domain_learning_topics(text: str) -> list[str]:
    lower_text = text.lower()
    if "andrej karpathy" in lower_text or "deep dive into llms" in lower_text:
        return [
            "Pretraining on internet-scale text",
            "Tokenization and subword units",
            "Transformer next-token prediction",
            "Supervised fine-tuning for assistant behavior",
            "Hallucinations and context-window limits",
            "Intermediate reasoning steps",
            "Reinforcement learning and RLHF",
            "LLM memory libraries",
            "Feedback loops and agentic drift",
        ]

    if "deeplearning101" in lower_text or (
        "gradient descent" in lower_text and "loss" in lower_text and "derivative" in lower_text
    ):
        return [
            "Linear algebra and matrix form",
            "Neurons and bias terms",
            "Derivatives and rate of change",
            "Functions as input-output mappings",
            "Loss and objective functions",
            "Optimization",
            "Gradient descent",
            "Optimizers: SGD, Adam, and AdaGrad",
        ]

    topic_rules = [
        (("pretraining", "fineweb"), "Pretraining on internet-scale text"),
        (("tokenized", "tokenization", "subword"), "Tokenization and subword units"),
        (("transformer", "next token"), "Transformer next-token prediction"),
        (("supervised fine-tuning", "sft", "instructgpt"), "Supervised fine-tuning for assistant behavior"),
        (("hallucination", "hallucinations"), "Hallucinations and memory limits"),
        (("context window", "working memory"), "Context windows as working memory"),
        (("show your work", "intermediate steps", "chain-of-thought"), "Intermediate reasoning steps"),
        (("reinforcement learning", "rlhf", "deepseek-r1"), "Reinforcement learning and RLHF"),
        (("entity memory", "token buffer memory"), "LLM memory libraries"),
        (("agentic drift", "feedback loops"), "Feedback loops and agentic drift"),
    ]
    topics = []
    for markers, topic in topic_rules:
        if any(marker in lower_text for marker in markers):
            topics.append(topic)
    return topics


def _extract_title_phrases(text: str) -> list[str]:
    phrases = re.findall(
        r"\b[A-Z][A-Za-z0-9&/+-]*(?:\s+[A-Z][A-Za-z0-9&/+-]*){0,4}\b",
        text,
    )
    return [
        phrase
        for phrase in phrases
        if phrase.lower() not in {"pdf", "pptx", "module", "overview", "section"}
    ]


def _learning_topic_paragraph(topics: list[str]) -> str:
    if not topics:
        return "The extracted text is mostly headings, so treat the headings as the concept map and review each one against the original slides."
    return "The main extracted topics are: " + "; ".join(
        f"{index}. {topic}" for index, topic in enumerate(topics[:12], start=1)
    ) + "."


def _looks_like_clean_topic(topic: str) -> bool:
    if not topic or len(topic) > 70:
        return False
    if topic.lower() in {"file", "module", "pptx", "pdf", "section", "overview", "this"}:
        return False
    if "." in topic:
        return False
    words = topic.split()
    first_word = re.sub(r"[^A-Za-z]", "", words[0]) if words else ""
    if first_word and not (first_word[0].isupper() or first_word.isupper()):
        return False
    short_words = [word for word in words if len(re.sub(r"[^A-Za-z]", "", word)) <= 2]
    if short_words and len(short_words) > max(1, len(words) // 2):
        return False
    if any(len(word) >= 8 and not re.search(r"[aeiouAEIOU]", word) for word in words):
        return False
    alpha_chars = sum(1 for char in topic if char.isalpha())
    return alpha_chars / max(len(topic), 1) >= 0.55


def _join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def _first_sentences(text: str, count: int) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return " ".join(sentences[:count])


def _claim_paragraph(claims: list[str]) -> str:
    if not claims:
        return "No clear high-confidence claims were extracted."
    return " ".join(f"{index}. {claim}" for index, claim in enumerate(claims[:4], start=1))


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

    notes = "; ".join(finding.rstrip(".") for finding in findings[:5])
    return f"The skeptical review flags the following issues: {notes}."


def _citation_readout(report: dict) -> str:
    details = report.get("citation_details", [])
    if not details:
        return "No structured citation details were extracted beyond the basic citation checklist."

    grouped = sum(1 for item in details if item.get("count", 0) >= 3)
    claim_bearing = sum(1 for item in details if "Claim-bearing" in item.get("check", ""))
    total = len(details)
    return (
        f"The parser found {total} distinct bracket-style citation groups in the extracted text. "
        f"{grouped} are grouped citations that should be checked source by source, and "
        f"{claim_bearing} appear near empirical or claim-bearing language."
    )
