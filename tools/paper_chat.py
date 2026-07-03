import re

from config import settings
from tools.chunker import chunk_text
from tools.llm_client import LLMClient
from tools.source_verifier import search_arxiv, search_wikipedia
from tools.text_cleaner import clean_display_text, clean_snippet_text
from tools.vector_store_factory import create_vector_store


MIN_DIRECT_ANSWER_SCORE = 0.30
MIN_SKEPTIC_SUPPORT_SCORE = 0.25
STRONG_SKEPTIC_SUPPORT_SCORE = 0.45


def build_paper_chat_store(paper_text: str, source_name: str):
    chunks = chunk_text(paper_text, settings.chunk_size, settings.chunk_overlap)
    store = create_vector_store()
    store.add_texts(
        chunks,
        metadata=[{"source": source_name, "chunk": index} for index in range(len(chunks))],
    )
    return store


def initialize_discussion_memory(report: dict | None = None) -> dict:
    report = report or {}
    sections = [
        section.get("title", "")
        for section in (report.get("narrative_review") or {}).get("sections", [])
        if section.get("title")
    ]
    topics = _known_topics(report)
    return {
        "current_section": "",
        "covered": [],
        "remaining": topics or sections,
        "important_entities": [],
        "last_question": "",
    }


def update_discussion_memory(memory: dict | None, question: str, response: dict, report: dict | None = None) -> dict:
    updated = dict(memory or initialize_discussion_memory(report))
    covered = list(updated.get("covered", []))
    entities = list(updated.get("important_entities", []))
    combined = f"{question} {response.get('answer', '')}"

    section = _infer_section(combined)
    if section:
        updated["current_section"] = section
        _append_unique(covered, section)

    for entity in _infer_entities(combined):
        _append_unique(entities, entity)
        _append_unique(covered, entity)
        if entity in {"PWC", "MT-Consistency", "CARG"}:
            updated["current_section"] = "Technical Contributions"

    known_topics = _known_topics(report or {})
    updated["covered"] = covered
    updated["remaining"] = [topic for topic in known_topics if topic not in covered]
    updated["important_entities"] = entities
    updated["last_question"] = clean_display_text(question)
    return updated


def answer_paper_question(
    question: str,
    store,
    report: dict | None = None,
    top_k: int = 4,
    discussion_memory: dict | None = None,
    web_search_enabled: bool = False,
    web_search_fn=None,
) -> dict:
    question = clean_display_text(question)
    report = report or {}

    routed_answer = _answer_from_report(question, report, discussion_memory)
    if routed_answer:
        return routed_answer

    matches = store.search(question, top_k=top_k) if store else []
    evidence = [
        {
            "score": float(match.get("score", 0)),
            "text": clean_snippet_text(match.get("text", ""), limit=420),
            "metadata": match.get("metadata", {}),
        }
        for match in matches
    ]

    best_score = evidence[0]["score"] if evidence else 0.0

    if best_score < MIN_DIRECT_ANSWER_SCORE and not _has_direct_topic_match(question, evidence):
        return _handle_insufficient_evidence(
            question,
            evidence,
            discussion_memory,
            web_search_enabled=web_search_enabled,
            web_search_fn=web_search_fn,
        )

    if not evidence:
        return {
            "answer": "I could not find relevant evidence in the uploaded paper for that question.",
            "evidence": [],
            "mode": "heuristic",
            "confidence": "Low",
            "confidence_reason": "No matching paper evidence was retrieved.",
            "sources": _sources([], discussion_memory),
        }

    answer, confidence, reason = _synthesize_answer(question, evidence)
    return {
        "answer": answer,
        "evidence": evidence,
        "mode": "heuristic",
        "confidence": confidence,
        "confidence_reason": reason,
        "sources": _sources(["Paper evidence"], discussion_memory),
    }


def skeptic_check_claim(claim: str, store, top_k: int = 5, report: dict | None = None) -> dict:
    claim = clean_display_text(claim)
    if not claim:
        return {
            "answer": "Paste a claim or proposed answer and I will check it against the paper.",
            "verdict": "Needs more evidence",
            "confidence": "Low",
            "confidence_reason": "No claim was provided.",
            "evidence": [],
            "follow_up_questions": [],
            "mode": "skeptic",
            "sources": ["Paper evidence"],
        }

    matches = store.search(claim, top_k=top_k) if store else []
    evidence = [
        {
            "score": float(match.get("score", 0)),
            "text": clean_snippet_text(match.get("text", ""), limit=420),
            "metadata": match.get("metadata", {}),
        }
        for match in matches
    ]
    best_score = evidence[0]["score"] if evidence else 0.0
    verdict, confidence, reason = _skeptic_verdict(claim, evidence)
    sources = ["Paper evidence"]
    if verdict in {"Unsupported", "Needs more evidence"} and report:
        report_evidence = _claim_report_evidence(claim, report)
        if report_evidence:
            evidence.insert(
                0,
                {
                    "score": MIN_SKEPTIC_SUPPORT_SCORE,
                    "text": report_evidence,
                    "metadata": {"source": "Final review"},
                },
            )
            best_score = max(best_score, MIN_SKEPTIC_SUPPORT_SCORE)
            verdict = "Partly supported"
            confidence = "Medium"
            reason = "The final review contains a related statement, but the claim should still be checked against the original evidence."
            sources.append("Final review")
    answer = _format_skeptic_answer(claim, verdict, confidence, reason, evidence)

    return {
        "answer": answer,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_reason": reason,
        "evidence": evidence,
        "follow_up_questions": _skeptic_follow_up_questions(claim, verdict, best_score),
        "mode": "skeptic",
        "sources": sources,
    }


def _skeptic_verdict(claim: str, evidence: list[dict]) -> tuple[str, str, str]:
    if not evidence:
        return "Needs more evidence", "Low", "No paper passages were retrieved for this claim."

    best_score = evidence[0]["score"]
    overlap = _claim_evidence_overlap(claim, evidence[0].get("text", ""))
    if overlap < 0.35:
        return "Unsupported", "Low", "The best passage is only loosely related to the claim."
    if _looks_contradicted(claim, evidence[:3]):
        return (
            "Contradicted",
            "Medium",
            "A retrieved passage appears to say the opposite or weakens the claim.",
        )
    if best_score >= STRONG_SKEPTIC_SUPPORT_SCORE:
        return "Supported", "High", "The paper has a strong matching passage for this claim."
    if best_score >= MIN_SKEPTIC_SUPPORT_SCORE:
        return "Partly supported", "Medium", "The paper has related evidence, but the claim may need tighter wording."
    return "Unsupported", "Low", "The retrieved paper evidence is weak for this claim."


def _format_skeptic_answer(
    claim: str,
    verdict: str,
    confidence: str,
    reason: str,
    evidence: list[dict],
) -> str:
    lines = [
        f"Verdict: {verdict}",
        f"Confidence: {confidence}",
        f"Why: {reason}",
    ]
    if evidence:
        lines.extend(
            [
                "",
                "Best paper evidence:",
                evidence[0]["text"],
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Best paper evidence:",
                "No matching paper evidence was found.",
            ]
        )
    if verdict in {"Unsupported", "Needs more evidence", "Partly supported"}:
        lines.extend(
            [
                "",
                "Skeptic note:",
                "Do not treat the claim as established until the paper shows direct evidence, scope, baselines, or citations.",
            ]
        )
    elif verdict == "Contradicted":
        lines.extend(
            [
                "",
                "Skeptic note:",
                "Revise the claim or inspect the cited passage before using it.",
            ]
        )
    return "\n\n".join(lines)


def _skeptic_follow_up_questions(claim: str, verdict: str, best_score: float) -> list[str]:
    questions = [
        "Which exact section, table, or experiment supports this?",
        "What assumptions must be true for this claim to hold?",
    ]
    if verdict in {"Unsupported", "Needs more evidence"} or best_score < MIN_SKEPTIC_SUPPORT_SCORE:
        questions.insert(0, "What direct paper evidence would prove this claim?")
    if _has_strong_language(claim):
        questions.append("Does the paper compare against enough baselines to justify the strong wording?")
    return questions


def _looks_contradicted(claim: str, evidence: list[dict]) -> bool:
    claim_lower = claim.lower()
    evidence_lower = " ".join(item.get("text", "") for item in evidence).lower()
    contradiction_pairs = [
        ("improves", ("does not improve", "fails to improve", "no improvement", "not improve")),
        ("outperforms", ("does not outperform", "fails to outperform", "underperforms")),
        ("significant", ("not significant", "insignificant", "no significant")),
        ("always", ("not always", "sometimes", "in some cases")),
        ("never", ("sometimes", "can", "may")),
    ]
    return any(
        claim_word in claim_lower and any(phrase in evidence_lower for phrase in opposing_phrases)
        for claim_word, opposing_phrases in contradiction_pairs
    )


def _has_strong_language(text: str) -> bool:
    return any(
        term in text.lower()
        for term in (
            "always",
            "never",
            "proves",
            "guarantees",
            "significant",
            "outperforms",
            "state-of-the-art",
            "best",
        )
    )


def _claim_evidence_overlap(claim: str, evidence_text: str) -> float:
    claim_terms = _important_skeptic_terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _important_skeptic_terms(evidence_text)
    return len(claim_terms.intersection(evidence_terms)) / len(claim_terms)


def _claim_report_evidence(claim: str, report: dict) -> str:
    narrative = report.get("narrative_review") or {}
    module_evidence = _claim_llm_module_evidence(claim, narrative)
    if module_evidence:
        return module_evidence

    candidates = []
    opening = clean_display_text(narrative.get("opening", ""))
    if opening:
        candidates.append(opening)
    for section in narrative.get("sections", []):
        title = clean_display_text(section.get("title", ""))
        body = clean_display_text(section.get("body", ""))
        if body:
            candidates.append(f"{title}: {body}" if title else body)

    claim_terms = _important_skeptic_terms(claim)
    if not claim_terms or not candidates:
        return ""

    chunks = []
    for candidate in candidates:
        candidate_chunks = _evidence_chunks(candidate)
        chunks.extend(candidate_chunks or [candidate])

    best_text = ""
    best_overlap = 0.0
    best_shared = 0
    for candidate in chunks:
        candidate_terms = _important_skeptic_terms(candidate)
        shared = len(claim_terms.intersection(candidate_terms))
        overlap = shared / len(claim_terms)
        shorter_tie = overlap == best_overlap and shared == best_shared and best_text and len(candidate) < len(best_text)
        if overlap > best_overlap or (overlap == best_overlap and shared > best_shared) or shorter_tie:
            best_text = candidate
            best_overlap = overlap
            best_shared = shared

    if best_overlap >= 0.35 or (best_shared >= 3 and best_overlap >= 0.25):
        return _brief(best_text, 520)
    return ""


def _claim_llm_module_evidence(claim: str, narrative: dict) -> str:
    if not _llm_reasoning_topics(narrative):
        return ""

    lower_claim = claim.lower()
    if any(term in lower_claim for term in ("hallucination", "hallucinate", "verification", "citation", "rag")):
        return (
            "The module covers reliability risks around hallucinations and explains that LLM outputs need grounding, "
            "RAG, citations, constrained output, or verification before being trusted."
        )
    if any(term in lower_claim for term in ("self-attention", "attention", "query", "key", "value")):
        return (
            "The module explains self-attention through Query, Key, and Value vectors and describes it as the core mechanism "
            "for connecting tokens with relevant context."
        )
    if any(term in lower_claim for term in ("prompt engineering", "prompt", "zero-shot", "few-shot")):
        return (
            "The module presents prompt engineering as a way to improve output quality, control format, and guide model behavior "
            "without fine-tuning."
        )
    if any(term in lower_claim for term in ("token", "embedding", "positional encoding")):
        return (
            "The module explains that LLMs process text as tokens, map tokens to embeddings, and add positional encoding so order "
            "is represented."
        )
    return ""


def _evidence_chunks(text: str) -> list[str]:
    text = clean_display_text(text)
    chunks = []
    chunks.extend(part.strip() for part in re.split(r"(?=\b\d+\.\s+)", text) if part.strip())
    chunks.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip())
    return [chunk for chunk in chunks if len(chunk) >= 40]


def _important_skeptic_terms(text: str) -> set[str]:
    stop_words = {
        "about",
        "against",
        "answer",
        "claim",
        "does",
        "from",
        "have",
        "into",
        "method",
        "paper",
        "proposed",
        "shows",
        "that",
        "this",
        "with",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        if token.lower() not in stop_words
    }


def _answer_from_report(question: str, report: dict, discussion_memory: dict | None = None) -> dict | None:
    lower_question = question.lower()
    narrative = report.get("narrative_review") or {}
    sections = narrative.get("sections", [])

    if any(
        term in lower_question
        for term in (
            "how should i go through",
            "how do i go through",
            "how should i read",
            "how do i read",
            "how should i study",
            "how do i study",
            "study this",
            "go through it",
            "go through this",
            "reading guide",
            "walk me through",
            "guide me",
        )
    ):
        guide = _llm_reasoning_reading_guide(narrative)
        if guide:
            return _report_response(guide, discussion_memory)
        guide = _build_reading_guide(narrative)
        if guide:
            return _report_response(guide, discussion_memory)

    if any(
        term in lower_question
        for term in (
            "main idea",
            "core idea",
            "summary",
            "overview",
            "explain the paper",
            "what does the paper say",
            "what does this paper say",
            "what does it say",
            "what is this paper about",
            "what is the paper about",
            "what does this document say",
            "what does the document say",
            "what is this document about",
            "what is the document about",
            "explain the document",
            "tell me about this paper",
            "tell me about the paper",
            "tell me about this document",
            "tell me about this",
        )
    ):
        core = _section_body(sections, "Core Idea")
        opening = narrative.get("opening", "")
        if core or opening:
            return _report_response(_build_overview_answer(narrative), discussion_memory)

    entity_answer = _answer_entity_from_report(lower_question, sections, discussion_memory)
    if entity_answer:
        return _report_response(entity_answer, discussion_memory)

    module_concept_answer = _answer_llm_module_concept(lower_question, narrative)
    if module_concept_answer:
        return _report_response(module_concept_answer, discussion_memory)

    if any(term in lower_question for term in ("contribution", "propose", "technical")):
        key_points = _llm_reasoning_key_points(narrative)
        if key_points:
            return _report_response(key_points, discussion_memory)
        body = _section_body(sections, "Technical Contributions") or _section_body(sections, "Key Claims")
        if body:
            return _report_response("Main contributions or key points: " + _brief(body, 700), discussion_memory)
        if sections:
            return _report_response(_build_overview_answer(narrative), discussion_memory)

    if any(term in lower_question for term in ("weakness", "limitation", "skeptical", "problem", "risk")):
        body = _section_body(sections, "Skeptical Notes")
        if body:
            return _report_response(_brief(body, 700), discussion_memory)
        if sections:
            return _report_response(_generic_weakness_answer(), discussion_memory)

    if any(term in lower_question for term in ("result", "finding", "experiment")):
        body = _section_body(sections, "Empirical Findings")
        if body:
            return _report_response(_brief(body, 700), discussion_memory)

    if _looks_like_broad_overview_question(lower_question) and sections:
        return _report_response(_build_overview_answer(narrative), discussion_memory)

    return None


def _report_response(answer: str, discussion_memory: dict | None) -> dict:
    return {
        "answer": answer,
        "evidence": [],
        "mode": "report",
        "confidence": "High",
        "confidence_reason": "The answer comes from the prepared final review.",
        "sources": _sources(["Final review"], discussion_memory),
    }


def _build_overview_answer(narrative: dict) -> str:
    friendly_overview = _llm_reasoning_overview(narrative)
    if friendly_overview:
        return friendly_overview

    sections = narrative.get("sections", [])
    opening = narrative.get("opening", "")
    core = _section_body(sections, "Core Idea")
    contributions = _section_body(sections, "Technical Contributions") or _section_body(sections, "Key Claims")
    findings = _section_body(sections, "Empirical Findings") or _section_body(sections, "Evidence Readout")
    skeptical = _section_body(sections, "Skeptical Notes")

    parts = []
    if opening or core:
        parts.append("Short answer: " + _brief(" ".join(part for part in (opening, core) if part), 650))
    if contributions:
        parts.append("What it contributes: " + _brief(contributions, 450))
    if findings:
        parts.append("What it finds: " + _brief(findings, 450))
    if skeptical:
        parts.append("What to be careful about: " + _brief(skeptical, 420))
    return "\n\n".join(parts)


def _build_reading_guide(narrative: dict) -> str:
    sections = narrative.get("sections", [])
    if not sections:
        return ""

    core = _section_body(sections, "Core Idea")
    contributions = _section_body(sections, "Technical Contributions") or _section_body(sections, "Key Claims")
    findings = _section_body(sections, "Empirical Findings") or _section_body(sections, "Evidence Readout")
    skeptical = _section_body(sections, "Skeptical Notes")

    steps = [
        "Go through it in this order:",
    ]
    if core:
        steps.append(f"1. Start with the core idea: {_brief(core, 320)}")
    if contributions:
        steps.append(f"2. Then check the technical contributions or key claims: {_brief(contributions, 320)}")
    if findings:
        steps.append(f"3. Read the empirical findings or evidence readout next: {_brief(findings, 320)}")
    if skeptical:
        steps.append(f"4. Finish with the skeptical notes: {_brief(skeptical, 320)}")
    steps.append(
        "After that, ask specific follow-up questions like 'What is PWC?', 'What is CARG?', "
        "'What evidence supports the results?', or 'Which claims need verification?'"
    )
    return "\n\n".join(steps)


def _generic_weakness_answer() -> str:
    return (
        "The final review does not list a dedicated weakness section. A skeptical reader should still check: "
        "whether the document gives citations for major claims, whether examples are enough to support the conclusion, "
        "whether limitations are stated clearly, and whether the material separates evidence from explanation."
    )


def _llm_reasoning_overview(narrative: dict) -> str:
    topics = _llm_reasoning_topics(narrative)
    if not topics:
        return ""

    lines = [
        (
            "Short answer: This document is a teaching module about LLMs as reasoning engines: "
            "how they process text, use attention, handle context, and use prompting techniques for reasoning-like tasks."
        ),
        "",
        "Main topics:",
    ]
    lines.extend(f"{index}. {topic}" for index, topic in enumerate(topics[:6], start=1))
    lines.append("")
    lines.append("Use it as a study guide for LLM mechanics and prompting, not as a formal research paper.")
    return "\n".join(lines)


def _llm_reasoning_key_points(narrative: dict) -> str:
    topics = _llm_reasoning_topics(narrative)
    if not topics:
        return ""

    lines = [
        "Main contributions or key points:",
        "1. It explains the basic LLM pipeline from tokens and embeddings to attention-based context understanding.",
        "2. It compares major model families and why scale, context length, and tool use matter.",
        "3. It introduces practical prompting patterns such as zero-shot prompting, structured prompts, and reasoning prompts.",
        "4. It warns that LLM outputs still need grounding, verification, and citation checks.",
    ]
    return "\n".join(lines)


def _llm_reasoning_reading_guide(narrative: dict) -> str:
    topics = _llm_reasoning_topics(narrative)
    if not topics:
        return ""

    return "\n\n".join(
        [
            "Go through it in this order:",
            "1. First learn the mechanics: tokens, embeddings, positional encoding, and self-attention.",
            "2. Then study context windows: system prompts, conversation history, retrieved memory, tools, and current observations.",
            "3. Next read the prompting section: prompt anatomy, zero-shot prompting, examples, and output format control.",
            "4. Then practice reasoning techniques such as chain-of-thought, decomposition, and reflection prompting.",
            "5. Finish with the safety/checking parts: hallucinations, citations, RAG, and verification.",
            "After each section, ask the app one specific question instead of asking for everything at once.",
        ]
    )


def _llm_reasoning_topics(narrative: dict) -> list[str]:
    text = _combined_narrative_text(narrative).lower()
    if not text:
        return []
    module_markers = ("module" in text and "section" in text) or "llms as reasoning engines" in text
    llm_markers = "llm" in text or "large language model" in text or "language models" in text
    mechanics_markers = any(
        marker in text
        for marker in (
            "self-attention",
            "token embeddings",
            "positional encoding",
            "chain-of-thought",
            "prompt engineering",
            "context window",
            "zero-shot",
        )
    )
    if not (module_markers and llm_markers and mechanics_markers):
        return []

    topic_checks = [
        (
            ("model family", "gpt", "claude", "gemini", "llama", "mistral"),
            "Modern LLM families and what each is known for.",
        ),
        (
            ("token embedding", "positional encoding", "tokenization"),
            "How text becomes tokens, embeddings, and ordered model inputs.",
        ),
        (
            ("self-attention", "attention(q", "query", "key", "value"),
            "How self-attention lets the model connect words and context.",
        ),
        (
            ("context window", "system prompt", "conversation history", "retrieved memory", "tool definitions"),
            "How context windows are divided between instructions, history, retrieved memory, tools, and observations.",
        ),
        (
            ("prompt engineering", "zero-shot", "few-shot", "anatomy of an effective prompt"),
            "Prompt engineering patterns for role, task, examples, input data, and output format.",
        ),
        (
            ("chain-of-thought", "reflection prompting", "prompt chaining", "decomposition"),
            "Reasoning techniques such as chain-of-thought, decomposition, chaining, and self-critique.",
        ),
        (
            ("hallucination", "rag", "citation", "verification"),
            "Reliability risks such as hallucinations and the need for grounding, citations, and verification.",
        ),
    ]
    topics = []
    for markers, description in topic_checks:
        if any(marker in text for marker in markers):
            topics.append(description)
    return topics


def _combined_narrative_text(narrative: dict) -> str:
    sections = narrative.get("sections", [])
    return " ".join(
        [narrative.get("opening", "")]
        + [section.get("body", "") for section in sections]
    )


def _brief(text: str, limit: int = 520) -> str:
    text = clean_display_text(text)
    if len(text) <= limit:
        return text
    return clean_snippet_text(text, limit=limit)


def _answer_entity_from_report(lower_question: str, sections: list[dict], discussion_memory: dict | None) -> str:
    definitions = _technical_definitions(sections)
    for aliases, label, body in definitions:
        if any(alias in lower_question for alias in aliases):
            context = _memory_context_sentence(label, discussion_memory)
            answer = f"{label}: {body}"
            return "\n\n".join(part for part in (context, answer) if part)
    return ""


def _answer_llm_module_concept(lower_question: str, narrative: dict) -> str:
    if not _llm_reasoning_topics(narrative):
        return ""

    concept_answers = [
        (
            ("self-attention", "attention"),
            (
                "Self-attention is the mechanism that lets an LLM connect each token with other relevant tokens in the context. "
                "In the module, it is described through Query, Key, and Value vectors: the model asks what each token is looking for, "
                "what other tokens contain, and what information should be gathered."
            ),
        ),
        (
            ("token", "tokens", "embedding", "embeddings", "positional encoding"),
            (
                "Tokens are the small text units an LLM processes. Embeddings turn those token IDs into dense vectors, and positional "
                "encoding adds order information so the transformer can tell where each token appears in the sequence."
            ),
        ),
        (
            ("context window", "context windows", "system prompt", "conversation history"),
            (
                "A context window is the limited space the model can read at once. The module breaks it into instructions, conversation "
                "history, retrieved memory or RAG context, tool definitions, tool results, and the current task."
            ),
        ),
        (
            ("prompt engineering", "zero-shot", "few-shot", "prompting"),
            (
                "Prompt engineering means shaping the input so the model knows the role, task, examples, input data, and output format. "
                "The module presents it as a practical way to improve results without retraining the model."
            ),
        ),
        (
            ("chain-of-thought", "chain of thought", "reasoning", "reflection", "prompt chaining"),
            (
                "The module presents reasoning techniques as prompting patterns: chain-of-thought asks for step-by-step reasoning, "
                "prompt chaining breaks work into stages, and reflection asks the model to critique and revise its answer."
            ),
        ),
        (
            ("hallucination", "hallucinations", "rag", "citation", "verification"),
            (
                "The module warns that LLMs can sound confident while being wrong. Grounding with RAG, citations, constrained output, "
                "and verification helps reduce that risk."
            ),
        ),
    ]
    for aliases, answer in concept_answers:
        if any(alias in lower_question for alias in aliases):
            return answer
    return ""


def _technical_definitions(sections: list[dict]) -> list[tuple[list[str], str, str]]:
    body = _section_body(sections, "Technical Contributions")
    definitions = []
    for paragraph in body.split("\n\n"):
        if ":" not in paragraph:
            continue
        label, detail = paragraph.split(":", 1)
        label = clean_display_text(label)
        detail = clean_display_text(detail)
        aliases = _aliases_for_label(label)
        definitions.append((aliases, label, detail))
    return definitions


def _aliases_for_label(label: str) -> list[str]:
    lower_label = label.lower()
    aliases = [lower_label]
    if "(" in label and ")" in label:
        acronym = label[label.find("(") + 1 : label.find(")")].strip()
        if acronym:
            aliases.append(acronym.lower())
    if "position-weighted consistency" in lower_label:
        aliases.append("pwc")
    if "mt-consistency" in lower_label:
        aliases.extend(["mt-consistency", "mt consistency"])
    if "confidence-aware response generation" in lower_label:
        aliases.append("carg")
    return list(dict.fromkeys(aliases))


def _memory_context_sentence(topic: str, discussion_memory: dict | None) -> str:
    if not discussion_memory:
        return ""
    covered = set(discussion_memory.get("covered", []))
    entities = set(discussion_memory.get("important_entities", []))
    current_section = discussion_memory.get("current_section", "")
    topic_entity = _friendly_entity_name(topic)
    if topic_entity in covered or topic_entity in entities:
        return f"Earlier in this discussion, we identified {topic_entity} under {current_section or 'the paper review'}."
    if current_section:
        return f"Discussion context: we are currently looking at {current_section}."
    return ""


def _friendly_entity_name(text: str) -> str:
    lower_text = text.lower()
    if "position-weighted consistency" in lower_text or "pwc" in lower_text:
        return "PWC"
    if "mt-consistency" in lower_text:
        return "MT-Consistency"
    if "confidence-aware response generation" in lower_text or "carg" in lower_text:
        return "CARG"
    return clean_display_text(text).split(":", 1)[0]


def _section_body(sections: list[dict], title: str) -> str:
    for section in sections:
        if section.get("title") == title:
            return section.get("body", "")
    return ""


def _looks_like_broad_overview_question(lower_question: str) -> bool:
    broad_phrases = (
        "what is this",
        "what does this document say",
        "what does the document say",
        "document about",
        "what is it saying",
        "what are they saying",
        "tell me more",
        "explain this",
        "explain it",
        "about this",
    )
    if any(phrase in lower_question for phrase in broad_phrases):
        specific_terms = (
            "pwc",
            "carg",
            "mt-consistency",
            "mt consistency",
            "citation",
            "reference",
            "table",
            "figure",
            "equation",
        )
        return not any(term in lower_question for term in specific_terms)
    return False


def _try_llm_answer(question: str, evidence: list[dict]) -> str:
    context = "\n\n".join(item["text"] for item in evidence[:4])
    prompt = (
        "Answer the user's question using only the paper excerpts below. "
        "If the excerpts are insufficient, say what is missing.\n\n"
        f"Question: {question}\n\nPaper excerpts:\n{context}"
    )
    try:
        return clean_display_text(LLMClient().complete(prompt))
    except Exception:
        return ""


def _synthesize_answer(question: str, evidence: list[dict]) -> tuple[str, str, str]:
    best = evidence[0]
    score = best["score"]
    confidence, reason = _confidence_from_score(score)

    answer = (
        f"{reason} I can ground the answer in this paper excerpt only:\n\n"
        f"{best['text']}\n\n"
        "I am not adding claims beyond that excerpt."
    )
    return answer, confidence, reason


def _handle_insufficient_evidence(
    question: str,
    evidence: list[dict],
    discussion_memory: dict | None,
    web_search_enabled: bool,
    web_search_fn=None,
) -> dict:
    if web_search_enabled:
        web_results, web_errors = _search_web_context(question, web_search_fn)
        valid_results = [result for result in web_results if not result.get("error")]
        if valid_results:
            return {
                "answer": _synthesize_web_answer(question, valid_results),
                "evidence": evidence,
                "web_results": web_results,
                "web_errors": web_errors,
                "mode": "web",
                "confidence": "Low",
                "confidence_reason": (
                    "The paper did not provide enough direct evidence, so this uses external context."
                ),
                "sources": _sources(["Web", "Paper evidence"], discussion_memory),
            }

        return {
            "answer": (
                "Web search was enabled, but I could not retrieve reliable external results. "
                "The paper evidence is also too weak for a grounded answer. Check your internet connection, "
                "try a more specific question, or use the External Source Check section."
            ),
            "evidence": evidence,
            "web_results": web_results,
            "web_errors": web_errors,
            "mode": "web_error",
            "confidence": "Low",
            "confidence_reason": "No usable paper or web evidence was available.",
            "sources": _sources(["Paper evidence"], discussion_memory),
        }

    return {
        "answer": (
            "I do not have enough evidence in the paper to answer that confidently. "
            "Enable web search to check external sources, or ask a more specific paper-focused question."
        ),
        "evidence": evidence,
        "mode": "guarded",
        "confidence": "Low",
        "confidence_reason": "The retrieved paper evidence was too weak for a grounded answer.",
        "sources": _sources(["Paper evidence"], discussion_memory),
    }


def _search_web_context(question: str, web_search_fn=None) -> tuple[list[dict], list[str]]:
    query = _web_query_from_question(question)
    if web_search_fn:
        results = web_search_fn(query)
        return results, [result.get("error", "") for result in results if result.get("error")]

    results = []
    results.extend(search_wikipedia(query, limit=2))
    results.extend(search_arxiv(query, limit=2))
    errors = [f"{result.get('source', 'Web')}: {result.get('error')}" for result in results if result.get("error")]
    return results, errors


def _synthesize_web_answer(question: str, web_results: list[dict]) -> str:
    lines = [
        "The paper did not provide enough direct evidence for this question. With web search enabled, I found this external context:",
    ]
    for index, result in enumerate(web_results[:4], start=1):
        title = clean_display_text(result.get("title", "Untitled source"))
        snippet = clean_snippet_text(result.get("snippet", ""), limit=260)
        url = result.get("url", "")
        source = result.get("source", "Web")
        lines.append(f"{index}. {source} - {title}: {snippet} {url}".strip())
    lines.append("Treat this as background context, not proof of the paper's claims.")
    return "\n\n".join(lines)


def _confidence_from_score(score: float) -> tuple[str, str]:
    if score >= 0.45:
        return "High", "The paper has a strong matching passage for this."
    if score >= 0.25:
        return "Medium", "The paper has a moderate matching passage for this."
    return "Low", "The paper evidence is limited."


def _web_query_from_question(question: str) -> str:
    query = clean_display_text(question).strip(" ?.!").lower()
    query = re.sub(
        r"\b(what|who|where|when|why|how|is|are|was|were|the|a|an|this|that|tell|me|about|explain|please)\b",
        " ",
        query,
    )
    query = re.sub(r"\s+", " ", query).strip()
    return query or clean_display_text(question).strip(" ?.!") or "research paper concept"


def _has_direct_topic_match(question: str, evidence: list[dict]) -> bool:
    entities = _infer_entities(question)
    if not entities or not evidence:
        return False
    evidence_text = " ".join(item.get("text", "") for item in evidence[:2]).lower()
    for entity in entities:
        if entity == "PWC" and ("pwc" in evidence_text or "position-weighted consistency" in evidence_text):
            return True
        if entity == "MT-Consistency" and ("mt-consistency" in evidence_text or "mt consistency" in evidence_text):
            return True
        if entity == "CARG" and ("carg" in evidence_text or "confidence-aware response generation" in evidence_text):
            return True
    return False


def _sources(source_names: list[str], discussion_memory: dict | None) -> list[str]:
    sources = list(source_names)
    if discussion_memory and (
        discussion_memory.get("current_section")
        or discussion_memory.get("covered")
        or discussion_memory.get("important_entities")
    ):
        sources.append("Discussion memory")
    return list(dict.fromkeys(sources))


def _known_topics(report: dict) -> list[str]:
    narrative = report.get("narrative_review") or {}
    sections = [
        section.get("title", "")
        for section in narrative.get("sections", [])
        if section.get("title")
    ]
    topics = sections[:]
    contributions = _section_body(narrative.get("sections", []), "Technical Contributions")
    if "Position-Weighted Consistency" in contributions:
        topics.append("PWC")
    if "MT-Consistency" in contributions:
        topics.append("MT-Consistency")
    if "Confidence-Aware Response Generation" in contributions:
        topics.append("CARG")
    return list(dict.fromkeys(topic for topic in topics if topic))


def _infer_section(text: str) -> str:
    lower_text = text.lower()
    if any(
        term in lower_text
        for term in (
            "core idea",
            "main idea",
            "overview",
            "what does the paper say",
            "what does this paper say",
            "tell me about this",
        )
    ):
        return "Core Idea"
    if any(term in lower_text for term in ("technical contribution", "pwc", "mt-consistency", "carg")):
        return "Technical Contributions"
    if any(term in lower_text for term in ("empirical", "experiment", "finding", "result")):
        return "Empirical Findings"
    if any(term in lower_text for term in ("weakness", "limitation", "skeptical", "risk")):
        return "Skeptical Notes"
    return ""


def _infer_entities(text: str) -> list[str]:
    lower_text = text.lower()
    entities = []
    if "pwc" in lower_text or "position-weighted consistency" in lower_text:
        entities.append("PWC")
    if "mt-consistency" in lower_text or "mt consistency" in lower_text:
        entities.append("MT-Consistency")
    if "carg" in lower_text or "confidence-aware response generation" in lower_text:
        entities.append("CARG")
    return entities


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)
