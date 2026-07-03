from __future__ import annotations

import re

from tools.text_cleaner import clean_display_text


CORE_SECTION_TITLES = {
    "overview",
    "core idea",
    "technical contributions",
    "empirical findings",
    "role of confidence and mitigation",
    "skeptical notes",
    "key claims",
    "evidence readout",
}


def build_workspace_sections(paper_text: str, report: dict) -> list[dict]:
    narrative = report.get("narrative_review") or {}
    sections = []

    opening = clean_display_text(narrative.get("opening", ""))
    if opening:
        sections.append(
            {
                "title": "Overview",
                "content": opening,
                "source": "Final review",
                "kind": "overview",
            }
        )

    for section in narrative.get("sections", []):
        title = clean_display_text(section.get("title", ""))
        body = clean_display_text(section.get("body", ""))
        if title and body:
            sections.append(
                {
                    "title": title,
                    "content": body,
                    "source": "Final review",
                    "kind": "review",
                }
            )

    for section in extract_paper_sections(paper_text):
        if not _already_has_section(sections, section["title"]):
            sections.append(section)

    return sections or [
        {
            "title": "Paper",
            "content": clean_display_text(paper_text[:2500]),
            "source": "Paper text",
            "kind": "paper",
        }
    ]


def extract_paper_sections(paper_text: str, max_sections: int = 12) -> list[dict]:
    lines = [clean_display_text(line) for line in paper_text.splitlines()]
    headings = []

    for index, line in enumerate(lines):
        if _looks_like_heading(line):
            headings.append((index, _normalize_heading(line)))

    sections = []
    for position, (start_index, title) in enumerate(headings[:max_sections]):
        end_index = headings[position + 1][0] if position + 1 < len(headings) else min(len(lines), start_index + 80)
        content = clean_display_text(" ".join(lines[start_index + 1 : end_index]))
        if content:
            sections.append(
                {
                    "title": title,
                    "content": content[:3500],
                    "source": "Paper text",
                    "kind": "paper",
                }
            )

    return sections


def build_section_study_payload(section: dict, understanding_level: str, reading_mode: str, report: dict) -> dict:
    title = section.get("title", "Selected section")
    content = clean_display_text(section.get("content", ""))
    level = understanding_level.lower()
    mode = reading_mode.lower()

    if mode == "evidence":
        explanation = "This mode stays close to the paper text and avoids extra teaching unless the paper supports it."
    elif mode == "skeptic":
        explanation = "Read this section by asking what is claimed, what evidence is shown, and what is still missing."
    else:
        explanation = _teaching_explanation(title, content, level, report)

    return {
        "paper_content": content,
        "simple_explanation": explanation,
        "real_world_example": _real_world_example(title, content, level, report),
        "why_it_matters": _why_it_matters(title, content, report),
        "related_concepts": related_concepts_for_section(title, report),
        "skeptical_questions": skeptical_questions_for_section(title),
        "verification_answers": verification_answers_for_section(section, report),
        "confidence": confidence_for_section(section),
    }


def build_learning_flowchart(report: dict, workspace_sections: list[dict] | None = None) -> list[dict]:
    workspace_sections = workspace_sections or []
    if _looks_like_llm_reasoning_context(report):
        return [
            _flow_node(
                "Input Text",
                "User question, document text, system instruction, and task goal enter the model.",
                "#2563eb",
                "This is everything the model is allowed to see for the current answer: your question, uploaded material, system rules, and any tool results.",
                ["user prompt", "system instruction", "document text"],
            ),
            _flow_node(
                "Tokens & Embeddings",
                "Text is split into tokens, then converted into dense vectors the model can process.",
                "#0891b2",
                "Tokenization breaks text into model-readable pieces. Embeddings turn those pieces into vectors, and positional encoding keeps their order visible.",
                ["tokenization", "embeddings", "positional encoding"],
            ),
            _flow_node(
                "Self-Attention",
                "Query, Key, and Value connections help the model decide which context matters.",
                "#7c3aed",
                "Self-attention lets each token compare itself with nearby and distant tokens, so the model can connect a question with relevant context.",
                ["query", "key", "value", "attention"],
            ),
            _flow_node(
                "Context Window",
                "Instructions, chat history, retrieved memory, tools, and observations compete for space.",
                "#db2777",
                "The context window is the model's working space. If important evidence is outside it, the model cannot use that evidence directly.",
                ["context window", "conversation history", "retrieved memory"],
            ),
            _flow_node(
                "Prompting & Reasoning",
                "Prompt structure, examples, decomposition, and reflection guide the answer.",
                "#ea580c",
                "Prompting controls role, task, examples, input data, and output format. Reasoning prompts break hard tasks into smaller checks.",
                ["prompt engineering", "chain-of-thought", "reflection"],
            ),
            _flow_node(
                "Verification",
                "Citations, RAG, external checks, and skeptical review reduce hallucination risk.",
                "#16a34a",
                "Verification asks whether the answer is grounded in the paper, citations, retrieved sources, or trusted external context.",
                ["RAG", "citations", "hallucinations", "external checks"],
            ),
        ]

    if _looks_like_consistency_report(report):
        return [
            _flow_node(
                "Correct First Answer",
                "Start with cases where the model initially answers correctly.",
                "#2563eb",
                "The paper focuses on whether a model preserves a correct answer after the conversation continues.",
                ["initial correctness", "answer persistence"],
            ),
            _flow_node(
                "Follow-up Pressure",
                "Users repeat, challenge, role-play, or introduce misleading suggestions.",
                "#c026d3",
                "Follow-up prompts simulate pressure: repetition, social agreement, role-play, tone changes, and subtle incorrect suggestions.",
                ["follow-up prompts", "role-play", "misleading suggestions"],
            ),
            _flow_node(
                "Consistency Test",
                "Check whether the model stays firm or changes without real evidence.",
                "#dc2626",
                "The central question is whether the answer changes because of real evidence or because the model is being swayed.",
                ["consistency", "sway resistance"],
            ),
            _flow_node(
                "PWC Metric",
                "Position-Weighted Consistency penalizes early sways and tracks recovery.",
                "#ea580c",
                "PWC gives extra weight to early turns, because an early wrong shift can damage the rest of the conversation.",
                ["PWC", "early sways", "recovery dynamics"],
            ),
            _flow_node(
                "MT-Consistency",
                "A benchmark stresses models across domains, difficulty, and prompt styles.",
                "#0891b2",
                "MT-Consistency is the benchmark layer: it creates multi-turn tests across task domains and difficulty levels.",
                ["benchmark", "multi-turn", "domains"],
            ),
            _flow_node(
                "CARG Mitigation",
                "Confidence-aware generation decides when to maintain or revise an answer.",
                "#16a34a",
                "CARG uses confidence signals to decide when a model should hold its answer and when revision is justified.",
                ["CARG", "confidence", "mitigation"],
            ),
            _flow_node(
                "Skeptical Check",
                "Review statistics, provenance, citations, and high-stakes validation before trusting the claim.",
                "#475569",
                "This is the reviewer layer: check author provenance, variance, statistical tests, citations, and domain validation.",
                ["statistics", "citations", "domain validation"],
            ),
        ]

    section_nodes = [
        _flow_node(
            section.get("title", "Section"),
            _first_sentence(section.get("content", "")),
            _flowchart_color(index),
            clean_display_text(section.get("content", ""))[:650],
            related_concepts_for_section(section.get("title", ""), report) or ["claim", "evidence", "limits"],
        )
        for index, section in enumerate(workspace_sections[:6])
        if section.get("title")
    ]
    return section_nodes or [
        _flow_node(
            "Read",
            "Identify the main claim in the document.",
            "#2563eb",
            "Start by naming the central claim and separating it from background explanation.",
            ["main claim", "scope"],
        ),
        _flow_node(
            "Check",
            "Compare the claim against evidence, citations, and limitations.",
            "#ea580c",
            "Look for direct evidence, citation support, missing tests, and unclear assumptions.",
            ["evidence", "citations", "limitations"],
        ),
        _flow_node(
            "Decide",
            "Use the claim only after its scope and support are clear.",
            "#16a34a",
            "Treat the claim as usable only inside the scope that the evidence actually supports.",
            ["decision", "confidence", "scope"],
        ),
    ]


def related_concepts_for_section(title: str, report: dict) -> list[str]:
    lower_title = title.lower()
    concepts = []
    if _looks_like_llm_reasoning_context(report, title):
        concept_map = {
            "overview": ["tokens", "embeddings", "self-attention", "prompting", "context windows"],
            "core idea": ["tokens", "embeddings", "self-attention", "prompting", "context windows"],
            "key claims": ["prompt engineering", "context management", "verification", "hallucinations"],
            "evidence": ["claim support", "retrieved evidence", "citation checks"],
            "skeptical": ["citations", "baseline comparisons", "reproducibility"],
        }
        for key, values in concept_map.items():
            if key in lower_title:
                return values[:6]
        return ["LLM reasoning", "attention", "prompting", "verification"]

    concept_map = {
        "overview": ["LLM reliability", "single-turn accuracy", "multi-turn consistency"],
        "core idea": ["accuracy", "consistency", "answer persistence"],
        "technical": ["PWC", "MT-Consistency", "CARG"],
        "contribution": ["PWC", "MT-Consistency", "CARG"],
        "empirical": ["adversarial follow-ups", "role-play", "confidence probing"],
        "result": ["accuracy", "sway resistance", "confidence"],
        "confidence": ["model uncertainty", "answer revision", "CARG"],
        "skeptical": ["provenance", "statistics", "external validation"],
        "limitation": ["statistical testing", "domain validation", "ethics review"],
    }
    for key, values in concept_map.items():
        if key in lower_title:
            concepts.extend(values)

    if not concepts and _looks_like_consistency_report(report):
        concepts.extend(["LLM consistency", "PWC", "CARG"])

    return list(dict.fromkeys(concepts))[:6]


def skeptical_questions_for_section(title: str) -> list[str]:
    lower_title = title.lower()
    if "technical" in lower_title or "contribution" in lower_title:
        return [
            "Is the metric clearly defined enough to reproduce?",
            "Does the benchmark cover realistic multi-turn interactions?",
            "Does the proposed mitigation preserve accuracy as well as consistency?",
        ]
    if "empirical" in lower_title or "result" in lower_title:
        return [
            "Are variance, confidence intervals, or statistical tests reported?",
            "Do the experiments compare against strong baselines?",
            "Could prompt choice or dataset filtering explain the result?",
        ]
    if "skeptical" in lower_title or "limitation" in lower_title:
        return [
            "Which claims need outside verification?",
            "What evidence is missing for high-stakes deployment?",
            "What would a domain expert need to approve?",
        ]
    return [
        "What claim is this section making?",
        "What evidence directly supports it?",
        "What assumption would change the conclusion?",
    ]


def verification_answers_for_section(section: dict, report: dict) -> list[dict]:
    questions = skeptical_questions_for_section(section.get("title", ""))
    return [
        {
            "question": question,
            "answer": answer_verification_question(question, section, report),
        }
        for question in questions
    ]


def answer_verification_question(question: str, section: dict, report: dict) -> str:
    title = section.get("title", "")
    content = clean_display_text(section.get("content", ""))
    lower_content = content.lower()
    lower_question = question.lower()

    if not content:
        return "Not enough extracted text is available in this section to answer this."

    if "metric clearly defined" in lower_question:
        if any(term in lower_content for term in ("metric", "position-weighted consistency", "pwc", "defined", "formula")):
            return (
                "Partly answered here: this section describes the metric or its purpose. "
                "For full reproducibility, check whether the paper also gives the exact formula, weighting rule, and evaluation procedure."
            )
        return "Not clearly answered in this section; look for the formal metric definition or formula."

    if "benchmark cover" in lower_question or "realistic multi-turn" in lower_question:
        if any(term in lower_content for term in ("benchmark", "multi-domain", "multi-difficulty", "adversarial", "follow-up", "naturalistic")):
            return (
                "Partly answered here: the section mentions benchmark coverage or multi-turn challenge design. "
                "It still needs checking whether the conversations reflect real user behavior."
            )
        return "Not enough evidence in this section to judge benchmark realism."

    if "preserve accuracy" in lower_question or "accuracy as well as consistency" in lower_question:
        if any(term in lower_content for term in ("accuracy", "without sacrificing", "preserve", "correctness", "stability")):
            return (
                "Partly answered here: the section discusses the accuracy/consistency tradeoff. "
                "To verify it, inspect the experiment tables and compare accuracy before and after the mitigation."
            )
        return "This section does not show enough evidence about whether accuracy is preserved."

    if "variance" in lower_question or "confidence intervals" in lower_question or "statistical tests" in lower_question:
        if any(term in lower_content for term in ("p-value", "confidence interval", "variance", "standard deviation", "statistical")):
            return "Answered in part: this section includes statistical reporting language that should be checked closely."
        return "Not visible here: this section does not clearly show p-values, confidence intervals, or variance reporting."

    if "strong baselines" in lower_question:
        if any(term in lower_content for term in ("baseline", "default", "gpt", "compare", "comparison")):
            return "Partly answered here: the section appears to compare against baseline or model variants."
        return "Not enough evidence in this section to confirm strong baselines."

    if "prompt choice" in lower_question or "dataset filtering" in lower_question:
        if any(term in lower_content for term in ("prompt", "dataset", "filter", "selected", "generated")):
            return "Possible risk: this section mentions prompts or dataset construction, so those choices could affect the result."
        return "This section does not provide enough detail to assess prompt or dataset-selection effects."

    if "outside verification" in lower_question:
        return "Claims needing outside verification include author provenance, citation support, statistical significance, and high-stakes domain validation."

    if "high-stakes deployment" in lower_question:
        if any(term in lower_content for term in ("healthcare", "education", "high-stakes", "domain", "ethics")):
            return "Partly answered here: the section mentions high-stakes relevance, but deployment would still need expert and ethics validation."
        return "The section does not provide enough direct evidence for high-stakes deployment readiness."

    if "domain expert" in lower_question:
        return "A domain expert would need clear evaluation data, error analysis, limitations, and evidence that the claims hold in the target domain."

    if "what claim" in lower_question:
        if _looks_like_llm_reasoning_context(report, title, content):
            return f"This section is mainly claiming: {_llm_module_main_claim(title, content)}"
        return f"This section appears to claim: {_first_sentence(content)}"

    if "what evidence" in lower_question:
        if _looks_like_llm_reasoning_context(report, title, content):
            module_evidence = _llm_module_direct_evidence(content)
            if module_evidence:
                return f"Direct module evidence: {module_evidence}"
        report_evidence = _best_report_evidence(report, content)
        if report_evidence:
            return f"Retrieved paper evidence: {report_evidence}"
        if section.get("source") == "Final review":
            return (
                "This selected section is a review summary, so it is not direct evidence by itself. "
                "Use the External Support button below for web/arXiv context, and use Advanced > Evidence readout "
                "for retrieved paper snippets."
            )
        return f"Direct paper text in this section: {_first_sentence(content)}"

    if "what assumption" in lower_question:
        if _looks_like_llm_reasoning_context(report, title, content):
            return (
                "A key assumption is that these simplified LLM mechanics and prompting practices apply to the "
                "model or tool being studied. Newer architectures, tool use, retrieval settings, or safety layers "
                "could change how well the section transfers."
            )
        concepts = related_concepts_for_section(title, report)
        if concepts:
            return f"A key assumption is that the section's discussion of {concepts[0]} transfers to the paper's broader setting."
        return "A key assumption is that the extracted section text is representative of the paper's broader argument."

    return "This question is only partly answerable from the selected section; check the paper evidence and final review before relying on it."


def confidence_for_section(section: dict) -> dict:
    source = section.get("source", "")
    content_length = len(section.get("content", ""))
    if source == "Final review":
        return {
            "level": "High",
            "reason": "This section comes from the prepared review summary.",
        }
    if content_length >= 800:
        return {
            "level": "Medium",
            "reason": "This section comes from extracted paper text with enough local context.",
        }
    return {
        "level": "Low",
        "reason": "This section has limited extracted text, so answers should stay cautious.",
    }


def _best_report_evidence(report: dict, section_content: str) -> str:
    evidence_items = report.get("evidence", [])
    if not evidence_items:
        return ""

    section_words = _important_words(section_content)
    best_text = ""
    best_score = -1.0
    for item in evidence_items:
        for match in item.get("matches", [])[:3]:
            text = clean_display_text(str(match.get("text", "")))
            overlap = len(section_words.intersection(_important_words(text)))
            score = float(match.get("score", 0)) + (overlap * 0.03)
            if text and score > best_score:
                best_score = score
                best_text = text

    if best_score < 0.12:
        return ""
    return _best_evidence_sentence(best_text, section_words)


def _llm_module_direct_evidence(content: str) -> str:
    lower_content = content.lower()
    evidence_points = []
    if any(term in lower_content for term in ("token embedding", "tokenization", "positional encoding")):
        evidence_points.append("it explains how text becomes tokens, embeddings, and ordered model inputs")
    if any(term in lower_content for term in ("self-attention", "query", "key", "value")):
        evidence_points.append("it describes self-attention through Query, Key, and Value connections")
    if any(term in lower_content for term in ("context window", "system prompt", "conversation history", "retrieved memory")):
        evidence_points.append("it breaks the context window into instructions, history, retrieved memory, tools, and current task information")
    if any(term in lower_content for term in ("prompt engineering", "zero-shot", "few-shot", "chain-of-thought")):
        evidence_points.append("it lists prompting patterns such as zero-shot, examples, output-format control, and reasoning prompts")
    if any(term in lower_content for term in ("hallucination", "rag", "citation", "verification")):
        evidence_points.append("it flags hallucinations and the need for grounding, citations, RAG, or verification")

    if not evidence_points:
        return ""
    return "; ".join(evidence_points[:4]) + "."


def _llm_module_main_claim(title: str, content: str) -> str:
    lower_text = f"{title} {content}".lower()
    lower_title = title.lower()
    if any(term in lower_title for term in ("overview", "core idea", "fundamental")):
        return (
            "LLMs process text by turning it into tokens and embeddings, using attention to connect context, and then "
            "predicting useful output under the guidance of prompts."
        )
    if "skeptical" in lower_text or "limitation" in lower_text:
        return (
            "LLM outputs can sound convincing while still needing citations, grounding, and verification before "
            "they are trusted."
        )
    if "evidence" in lower_text or "claim" in lower_text:
        return (
            "claims about LLM behavior should be tied to concrete examples, retrieved evidence, citations, or "
            "external checks."
        )
    if "prompt" in lower_text or "chain-of-thought" in lower_text:
        return (
            "prompt design can guide an LLM's role, task, examples, reasoning steps, and output format without "
            "changing the model weights."
        )
    if "context window" in lower_text or "retrieved memory" in lower_text:
        return (
            "an LLM can only use information that fits in its current context, including instructions, chat history, "
            "retrieved memory, tool information, and the current task."
        )
    return (
        "LLMs process text by turning it into tokens and embeddings, using attention to connect context, and then "
        "predicting useful output under the guidance of prompts."
    )


def build_knowledge_graph(report: dict) -> list[tuple[str, str]]:
    if _looks_like_consistency_report(report):
        return [
            ("Reliability", "Accuracy"),
            ("Reliability", "Consistency"),
            ("Consistency", "PWC"),
            ("Consistency", "MT-Consistency"),
            ("Consistency", "CARG"),
            ("Consistency", "Confidence"),
            ("Confidence", "Sway resistance"),
            ("Sway resistance", "Misleading follow-ups"),
        ]

    concepts = []
    for claim in report.get("claims", [])[:5]:
        if isinstance(claim, dict):
            concept = _short_concept(claim.get("claim", ""))
            if concept:
                concepts.append(("Paper", concept))
    return concepts or [("Paper", "Claims"), ("Paper", "Evidence"), ("Paper", "Limitations")]


def format_knowledge_graph(edges: list[tuple[str, str]]) -> str:
    children: dict[str, list[str]] = {}
    for parent, child in edges:
        children.setdefault(parent, [])
        children.setdefault(child, [])
        if child not in children[parent]:
            children[parent].append(child)

    roots = [parent for parent in children if not any(parent in child_list for child_list in children.values())]
    lines = []
    for root in roots[:3]:
        _append_tree_lines(lines, root, children, depth=0, seen=set())
    return "\n".join(lines)


def search_workspace(sections: list[dict], query: str, notes: dict | None = None) -> list[dict]:
    query = clean_display_text(query).lower()
    if not query:
        return []

    results = []
    for section in sections:
        haystack = f"{section.get('title', '')} {section.get('content', '')}".lower()
        if query in haystack:
            results.append(
                {
                    "title": section.get("title", ""),
                    "source": section.get("source", ""),
                    "snippet": _query_snippet(section.get("content", ""), query),
                }
            )

    for title, note in (notes or {}).items():
        if query in str(note).lower():
            results.append(
                {
                    "title": f"Note: {title}",
                    "source": "My notes",
                    "snippet": clean_display_text(str(note))[:240],
                }
            )
    return results[:8]


def completed_progress(completed_sections: list[str], sections: list[dict]) -> dict:
    total = max(len(sections), 1)
    completed = len([section for section in sections if section.get("title") in set(completed_sections)])
    return {
        "completed": completed,
        "total": total,
        "ratio": completed / total,
    }


def _teaching_explanation(title: str, content: str, level: str, report: dict) -> str:
    if "beginner" in level or "7" in level:
        if _looks_like_llm_reasoning_context(report, title, content):
            return (
                "Think of this section as the map of how an LLM turns text into an answer: it breaks text into tokens, "
                "represents them as embeddings, uses attention to connect context, and then follows prompts to generate output."
            )
        if _looks_like_consistency_context(report, title, content):
            return (
                "Think of the model like a student who gives a correct answer, then gets pressured to change it. "
                f"This section, {title}, helps explain whether the model stays with the correct answer or gets swayed."
            )
        return f"This section is mainly saying: {_first_sentence(content)}"

    if "exam" in level:
        return (
            f"For exam preparation, remember the section's main point first: {_first_sentence(content)} "
            "Then connect it to the paper's method, evidence, and limitations."
        )

    if "researcher" in level:
        return (
            f"Researcher lens: this section should be read for its claim, operational definition, evidence quality, "
            f"and possible threats to validity. Main signal: {_first_sentence(content)}"
        )

    return f"In plain terms, this section says: {_first_sentence(content)}"


def _real_world_example(title: str, content: str, level: str, report: dict) -> str:
    lower_title = title.lower()
    lower_content = content.lower()
    if _looks_like_llm_reasoning_context(report, title, content):
        if "skeptical" in lower_title or "limitation" in lower_title:
            return (
                "If a chatbot gives a confident explanation with no citation, a reviewer should ask where the claim came from, "
                "whether the example proves it, and how the answer was verified."
            )
        if "evidence" in lower_title or "claim" in lower_title:
            return (
                "If the module says prompt engineering improves accuracy, the evidence section should point to the exact slide, "
                "example, paper, or benchmark that supports that statement."
            )
        return (
            "When you ask an AI to summarize a PDF, it first converts your text into tokens, uses attention to connect related words, "
            "and follows your prompt instructions to produce the answer. This section explains those moving parts."
        )
    if _looks_like_consistency_context(report, title, content):
        if "pwc" in lower_title or "technical" in lower_title or "contribution" in lower_title:
            return (
                "Imagine a medical assistant first gives the correct advice, then the patient says, "
                "'Are you sure? My friend said the opposite.' PWC, MT-Consistency, and CARG are ways to measure or reduce "
                "that kind of answer-shifting."
            )
        if "empirical" in lower_title or "result" in lower_title or "experiment" in lower_title:
            return (
                "A test can start with a model answering correctly, then add misleading follow-up prompts, role-play, "
                "or social pressure. If the answer changes for the wrong reason, the model is accurate once but unreliable over a conversation."
            )
        if "confidence" in lower_title:
            return (
                "A careful tutor may say 'I am not sure' when evidence is weak, but should not abandon a correct answer just because "
                "the student asks again with more confidence."
            )
        if "skeptical" in lower_title or "limitation" in lower_title:
            return (
                "Before using this idea in healthcare or education, a reviewer would ask for author provenance, statistical tests, "
                "and domain-expert validation instead of trusting the claim on wording alone."
            )
        if any(term in lower_title or term in lower_content for term in ("core", "overview", "main claim", "answer persistence")):
            return (
                "If an AI correctly says 5 + 5 = 10, then the user says 'Are you sure? My teacher says 11,' "
                "a reliable model should stay with 10 unless new evidence appears. This paper studies that behavior in longer conversations."
            )
        return (
            "If a model gives a correct answer, then faces repeated follow-up questions, the practical question is whether it stays correct "
            "or gets pushed into changing its answer without real evidence."
        )
    return (
        "If someone wants to use this section in a summary or decision, they should check the claim "
        "against the paper's examples, evidence, and stated limits."
    )


def _why_it_matters(title: str, content: str, report: dict) -> str:
    lower_title = title.lower()
    if _looks_like_llm_reasoning_context(report, title, content):
        if "skeptical" in lower_title or "limitation" in lower_title:
            return "This matters because LLM outputs can sound correct even when they need citations, grounding, or clearer limits."
        if "evidence" in lower_title or "claim" in lower_title:
            return "This matters because learning LLMs is safer when every big claim is connected to examples, citations, or checks."
        return (
            "This matters because understanding tokens, attention, context, and prompting makes LLM behavior less mysterious "
            "and helps you use these systems more reliably."
        )
    if _looks_like_consistency_context(report, title, content):
        if "skeptical" in lower_title or "limitation" in lower_title:
            return (
                "This matters because high-stakes uses need more than a plausible claim: they need provenance, statistics, "
                "domain validation, and clear limits."
            )
        if "technical" in lower_title or "contribution" in lower_title:
            return (
                "This matters because PWC, MT-Consistency, and CARG turn the broad idea of reliability into something that can be "
                "measured, tested, and compared."
            )
        if "empirical" in lower_title or "result" in lower_title:
            return (
                "This matters because high single-turn accuracy can hide a model that becomes unstable when users challenge it over "
                "multiple turns."
            )
        if "confidence" in lower_title:
            return (
                "This matters because confidence helps explain when a model is likely to hold a correct answer and when it may be "
                "vulnerable to misleading pressure."
            )
        return (
            "This matters because people rarely ask AI only one isolated question; real reliability means staying correct through "
            "follow-ups, corrections, tone shifts, and pressure."
        )
    if "skeptical" in lower_title or "limitation" in lower_title:
        return "This matters because weak validation can make a strong-sounding paper less dependable in real use."
    if "technical" in lower_title or "contribution" in lower_title:
        return "This matters because the paper's value depends on whether these tools are clear, useful, and reproducible."
    if "empirical" in lower_title or "result" in lower_title:
        return "This matters because results decide whether the idea actually works beyond a persuasive explanation."
    return "This matters because it shows what should be verified before trusting or reusing the section's claim."


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    normalized = line.strip()
    lower = normalized.lower().rstrip(".")
    named_headings = {
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "limitations",
        "future work",
        "conclusion",
        "references",
    }
    if lower in named_headings:
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,:()/-]{3,80}$", normalized))


def _normalize_heading(line: str) -> str:
    line = re.sub(r"^\d+(?:\.\d+)*\s+", "", line).strip()
    return clean_display_text(line).title()


def _already_has_section(sections: list[dict], title: str) -> bool:
    normalized = title.lower()
    return normalized in CORE_SECTION_TITLES or any(section.get("title", "").lower() == normalized for section in sections)


def _looks_like_consistency_report(report: dict) -> bool:
    return _looks_like_consistency_context(report)


def _looks_like_llm_reasoning_context(report: dict, title: str = "", content: str = "") -> bool:
    text = " ".join(
        [
            title,
            content,
            report.get("summary", ""),
            (report.get("narrative_review") or {}).get("opening", ""),
            " ".join(section.get("body", "") for section in (report.get("narrative_review") or {}).get("sections", [])),
        ]
    ).lower()
    module_marker = "llms as reasoning engines" in text or ("module" in text and "large language models" in text)
    mechanics_marker = any(
        marker in text
        for marker in (
            "token embedding",
            "positional encoding",
            "self-attention",
            "context window",
            "prompt engineering",
            "chain-of-thought",
            "hallucination",
        )
    )
    return module_marker and mechanics_marker


def _looks_like_consistency_context(report: dict, title: str = "", content: str = "") -> bool:
    lower_title = title.lower()
    text = " ".join(
        [
            title,
            content,
            report.get("summary", ""),
            (report.get("narrative_review") or {}).get("opening", ""),
            " ".join(section.get("body", "") for section in (report.get("narrative_review") or {}).get("sections", [])),
        ]
    ).lower()
    has_consistency_marker = any(
        term in text
        for term in (
            "position-weighted consistency",
            "mt-consistency",
            "carg",
            "multi-turn",
            "consistent answer",
            "consistency",
            "answer persistence",
            "follow-up",
            "follow-up prompt",
            "follow-up question",
            "single-turn accuracy",
            "stays firm",
            "fickle",
            "swayed",
            "sway resistance",
        )
    )
    if has_consistency_marker:
        return True
    if "core idea" in lower_title and any(term in text for term in ("accuracy", "consistency", "reliability", "answer")):
        return True
    return False


def _first_sentence(text: str) -> str:
    text = clean_display_text(text)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    return sentences[0] if sentences else "the extracted text is too limited for a clean explanation."


def _short_concept(text: str) -> str:
    text = clean_display_text(text)
    words = text.split()
    return " ".join(words[:6]).rstrip(".,;:") if words else ""


def _flowchart_color(index: int) -> str:
    palette = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ea580c", "#16a34a"]
    return palette[index % len(palette)]


def _flow_node(title: str, body: str, color: str, topic_body: str, keywords: list[str]) -> dict:
    return {
        "title": clean_display_text(title),
        "body": clean_display_text(body),
        "color": color,
        "topic_id": _topic_id(title),
        "topic_title": f"Topic: {clean_display_text(title)}",
        "topic_body": clean_display_text(topic_body),
        "keywords": [clean_display_text(keyword) for keyword in keywords if keyword],
    }


def _topic_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", clean_display_text(title).lower()).strip("-")
    return slug or "topic"


def _query_snippet(text: str, query: str) -> str:
    clean_text = clean_display_text(text)
    lower_text = clean_text.lower()
    index = lower_text.find(query)
    if index < 0:
        return clean_text[:240]
    start = max(index - 90, 0)
    end = min(index + 180, len(clean_text))
    prefix = "..." if start else ""
    suffix = "..." if end < len(clean_text) else ""
    return f"{prefix}{clean_text[start:end]}{suffix}"


def _best_evidence_sentence(text: str, section_words: set[str]) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_display_text(text))
        if sentence.strip()
    ]
    if not sentences:
        return clean_display_text(text)[:420].rsplit(" ", 1)[0].rstrip() + "..."

    best_sentence = max(
        sentences,
        key=lambda sentence: len(section_words.intersection(_important_words(sentence))),
    )
    if len(best_sentence) <= 420:
        return best_sentence
    return best_sentence[:420].rsplit(" ", 1)[0].rstrip() + "..."


def _important_words(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "paper",
        "report",
        "section",
        "large",
        "language",
        "models",
    }
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z-]{3,}", clean_display_text(text))
        if word.lower() not in stopwords
    }


def _append_tree_lines(lines: list[str], node: str, children: dict[str, list[str]], depth: int, seen: set[str]) -> None:
    if node in seen:
        return
    seen.add(node)
    indent = "  " * depth
    lines.append(f"{indent}- {node}")
    for child in children.get(node, []):
        _append_tree_lines(lines, child, children, depth + 1, seen)
