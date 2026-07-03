from tools.research_workspace import (
    answer_verification_question,
    build_knowledge_graph,
    build_learning_flowchart,
    build_section_study_payload,
    build_workspace_sections,
    completed_progress,
    extract_paper_sections,
    format_knowledge_graph,
    search_workspace,
)


def test_extract_paper_sections_detects_numbered_headings() -> None:
    paper_text = """Abstract
This paper studies reliability.

1 Introduction
Large language models are evaluated.

2 Method
We introduce a metric.
"""

    sections = extract_paper_sections(paper_text)

    assert [section["title"] for section in sections][:3] == ["Abstract", "Introduction", "Method"]
    assert "reliability" in sections[0]["content"]


def test_build_workspace_sections_includes_review_overview() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper studies multi-turn consistency.",
            "sections": [{"title": "Core Idea", "body": "Reliability needs stable answers."}],
        }
    }

    sections = build_workspace_sections("1 Introduction\nPaper text.", report)

    assert sections[0]["title"] == "Overview"
    assert sections[0]["source"] == "Final review"
    assert any(section["title"] == "Core Idea" for section in sections)


def test_section_study_payload_changes_for_beginner() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper studies multi-turn consistency and CARG.",
            "sections": [],
        }
    }
    section = {"title": "Core Idea", "content": "Models should keep correct answers stable.", "source": "Final review"}

    payload = build_section_study_payload(section, "Beginner", "Explain", report)

    assert "student" in payload["simple_explanation"] or "model" in payload["simple_explanation"]
    assert payload["confidence"]["level"] == "High"


def test_consistency_core_idea_uses_specific_study_guidance() -> None:
    report = {"narrative_review": {"opening": "", "sections": []}}
    section = {
        "title": "Core Idea",
        "content": "The paper asks whether LLMs maintain consistent answers over multi-turn follow-up conversations.",
        "source": "Final review",
    }

    payload = build_section_study_payload(section, "Beginner", "Explain", report)

    assert "5 + 5 = 10" in payload["real_world_example"]
    assert "follow-ups" in payload["why_it_matters"]
    assert "generic" not in payload["real_world_example"].lower()


def test_core_idea_with_accuracy_wording_does_not_use_generic_fallback() -> None:
    report = {"narrative_review": {"opening": "", "sections": []}}
    section = {
        "title": "Core Idea",
        "content": (
            "The paper argues that reliability is not just single-turn accuracy. "
            "It asks whether an LLM stays firm when its first answer is correct."
        ),
        "source": "Final review",
    }

    payload = build_section_study_payload(section, "Beginner", "Explain", report)

    assert "usable outside the paper" not in payload["real_world_example"]
    assert "supporting details" not in payload["why_it_matters"]
    assert "5 + 5 = 10" in payload["real_world_example"]


def test_consistency_technical_section_explains_metric_value() -> None:
    report = {"narrative_review": {"opening": "", "sections": []}}
    section = {
        "title": "Technical Contributions",
        "content": "The paper introduces Position-Weighted Consistency, MT-Consistency, and CARG.",
        "source": "Final review",
    }

    payload = build_section_study_payload(section, "Researcher", "Explain", report)

    assert "PWC" in payload["real_world_example"]
    assert "measured" in payload["why_it_matters"]


def test_llm_module_overview_uses_specific_study_guidance() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines. Token Embeddings, Positional Encoding, Self-Attention, Prompt Engineering.",
            "sections": [],
        }
    }
    section = {
        "title": "Overview",
        "content": "Large Language Models process text with token embeddings and self-attention.",
        "source": "Final review",
    }

    payload = build_section_study_payload(section, "Beginner", "Explain", report)

    assert "tokens" in payload["simple_explanation"]
    assert "converts your text into tokens" in payload["real_world_example"]
    assert "usable outside the paper" not in payload["real_world_example"]
    assert "supporting details" not in payload["why_it_matters"]
    assert "self-attention" in payload["related_concepts"]


def test_build_learning_flowchart_for_llm_module() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Positional Encoding, Self-Attention, Prompt Engineering, Chain-of-Thought, hallucination, RAG, citations, and verification.",
                }
            ],
        }
    }

    nodes = build_learning_flowchart(report)

    titles = [node["title"] for node in nodes]
    assert titles[:3] == ["Input Text", "Tokens & Embeddings", "Self-Attention"]
    assert titles[-1] == "Verification"
    assert all(node["color"].startswith("#") for node in nodes)
    assert all(node["topic_id"] for node in nodes)
    assert all(node["topic_body"] for node in nodes)
    assert len({node["topic_id"] for node in nodes}) == len(nodes)
    assert "embeddings" in nodes[1]["keywords"]


def test_build_learning_flowchart_for_consistency_report() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper introduces Position-Weighted Consistency, MT-Consistency, CARG, multi-turn follow-up pressure.",
            "sections": [],
        }
    }

    nodes = build_learning_flowchart(report)

    titles = [node["title"] for node in nodes]
    assert "PWC Metric" in titles
    assert "CARG Mitigation" in titles
    assert titles[0] == "Correct First Answer"
    assert all(node["topic_id"] for node in nodes)
    assert "PWC" in nodes[3]["keywords"]


def test_evidence_mode_payload_includes_verification_answers() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper studies multi-turn consistency and CARG.",
            "sections": [],
        }
    }
    section = {
        "title": "Technical Contributions",
        "content": "Position-Weighted Consistency (PWC) is a metric. CARG improves stability without sacrificing accuracy.",
        "source": "Final review",
    }

    payload = build_section_study_payload(section, "Researcher", "Evidence", report)

    assert payload["verification_answers"]
    assert payload["verification_answers"][0]["question"]
    assert payload["verification_answers"][0]["answer"]


def test_answer_verification_question_identifies_missing_statistics() -> None:
    section = {
        "title": "Empirical Findings",
        "content": "The experiments show models can be swayed by misleading follow-up prompts.",
        "source": "Final review",
    }

    answer = answer_verification_question(
        "Are variance, confidence intervals, or statistical tests reported?",
        section,
        {},
    )

    assert "does not clearly show" in answer


def test_answer_verification_question_does_not_treat_summary_as_evidence() -> None:
    section = {
        "title": "Overview",
        "content": "The report analyzes how well large language models maintain consistent answers.",
        "source": "Final review",
    }

    answer = answer_verification_question("What evidence directly supports it?", section, {})

    assert "review summary" in answer
    assert "not direct evidence" in answer


def test_answer_verification_question_uses_retrieved_report_evidence() -> None:
    section = {
        "title": "Overview",
        "content": "The report analyzes multi-turn consistency.",
        "source": "Final review",
    }
    report = {
        "evidence": [
            {
                "matches": [
                    {
                        "score": 0.42,
                        "text": "Large Language Models require consistency in multi-turn interactions across misleading follow-up prompts.",
                    }
                ]
            }
        ]
    }

    answer = answer_verification_question("What evidence directly supports it?", section, report)

    assert "Retrieved paper evidence" in answer
    assert "multi-turn interactions" in answer


def test_llm_module_evidence_answer_uses_module_specific_support() -> None:
    section = {
        "title": "Overview",
        "content": (
            "MODULE 4 LLMs as Reasoning Engines. Token Embeddings and Positional Encoding explain inputs. "
            "Self-Attention uses Query, Key, and Value vectors. Prompt Engineering controls output format."
        ),
        "source": "Final review",
    }
    report = {
        "narrative_review": {
            "opening": section["content"],
            "sections": [],
        },
        "evidence": [
            {
                "matches": [
                    {
                        "score": 0.91,
                        "text": "Third-party tools, plugins, or MCP servers could be compromised and should be validated.",
                    }
                ]
            }
        ],
    }

    answer = answer_verification_question("What evidence directly supports it?", section, report)

    assert "Direct module evidence" in answer
    assert "tokens" in answer
    assert "Self-Attention" in answer or "self-attention" in answer
    assert "Third-party tools" not in answer


def test_llm_module_verification_claim_answer_is_concise() -> None:
    section = {
        "title": "Overview",
        "content": (
            "MODULE 4 LLMs as Reasoning Engines How Large Language Models Think, Reason, and Generate "
            "SECTION 1 Fundamentals of Large Language Models How LLMs process text, predict tokens, "
            "and develop reasoning The Modern LLM Landscape Model Family Developer Key Characteristics "
            "Token Embeddings and Positional Encoding Self-Attention Prompt Engineering."
        ),
        "source": "Final review",
    }
    report = {"narrative_review": {"opening": section["content"], "sections": []}}

    claim_answer = answer_verification_question("What claim is this section making?", section, report)
    assumption_answer = answer_verification_question("What assumption would change the conclusion?", section, report)

    assert "turning it into tokens" in claim_answer
    assert len(claim_answer) < 300
    assert "Model Family Developer Key Characteristics" not in claim_answer
    assert "simplified LLM mechanics" in assumption_answer


def test_answer_verification_question_returns_complete_evidence_sentence() -> None:
    section = {
        "title": "Technical Contributions",
        "content": "The paper proposes PWC, MT-Consistency, and CARG.",
        "source": "Final review",
    }
    report = {
        "evidence": [
            {
                "matches": [
                    {
                        "score": 0.38,
                        "text": (
                            "To address these gaps, our research introduces three key contributions: "
                            "the Position-Weighted Consistency (PWC) metric, emphasizing early-stage stability "
                            "and recovery dynamics; the MT-Consistency benchmark, an extensive dataset for "
                            "evaluating multi-turn consistency; and Confidence-Aware Response Generation."
                        ),
                    }
                ]
            }
        ]
    }

    answer = answer_verification_question("What evidence directly supports it?", section, report)

    assert answer.endswith("Generation.")
    assert "evaluati" not in answer[-12:]


def test_workspace_search_finds_sections_and_notes() -> None:
    sections = [{"title": "Technical Contributions", "content": "PWC measures consistency.", "source": "Final review"}]
    notes = {"Core Idea": "Remember CARG as a mitigation method."}

    results = search_workspace(sections, "CARG", notes)

    assert results[0]["title"] == "Note: Core Idea"
    assert results[0]["source"] == "My notes"


def test_completed_progress_counts_selected_sections() -> None:
    sections = [{"title": "Overview"}, {"title": "Method"}]

    progress = completed_progress(["Overview"], sections)

    assert progress["completed"] == 1
    assert progress["total"] == 2
    assert progress["ratio"] == 0.5


def test_knowledge_graph_formats_consistency_concepts() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper introduces Position-Weighted Consistency, MT-Consistency, and CARG.",
            "sections": [],
        }
    }

    graph_text = format_knowledge_graph(build_knowledge_graph(report))

    assert "Reliability" in graph_text
    assert "PWC" in graph_text
