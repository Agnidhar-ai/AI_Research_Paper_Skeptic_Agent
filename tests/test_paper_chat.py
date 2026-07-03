from tools.paper_chat import (
    answer_paper_question,
    build_paper_chat_store,
    initialize_discussion_memory,
    skeptic_check_claim,
    update_discussion_memory,
)


def test_paper_chat_answers_from_report_sections() -> None:
    report = {
        "narrative_review": {
            "opening": "This paper studies consistency.",
            "sections": [{"title": "Core Idea", "body": "Reliability requires stable multi-turn answers."}],
        }
    }

    answer = answer_paper_question("What is the main idea?", None, report)

    assert "stable multi-turn answers" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_retrieves_evidence() -> None:
    text = "Confidence-Aware Response Generation improves response stability in multi-turn settings."
    store = build_paper_chat_store(text, "demo.pdf")

    answer = answer_paper_question("How does CARG help?", store)

    assert "Confidence-Aware Response Generation" in answer["answer"]
    assert "I am not adding claims beyond that excerpt" in answer["answer"]
    assert answer["evidence"]


def test_paper_chat_answers_broad_question_from_report() -> None:
    report = {
        "narrative_review": {
            "opening": "The paper reviews LLM consistency.",
            "sections": [
                {"title": "Core Idea", "body": "Models should keep correct answers stable."},
                {"title": "Technical Contributions", "body": "It introduces PWC and CARG."},
                {"title": "Empirical Findings", "body": "Models can be swayed by follow-ups."},
                {"title": "Skeptical Notes", "body": "Statistics need stronger reporting."},
            ],
        }
    }

    answer = answer_paper_question("what does the paper say", None, report)

    assert "Short answer" in answer["answer"]
    assert "PWC and CARG" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_answers_this_paper_wording_from_report() -> None:
    report = {
        "narrative_review": {
            "opening": "This paper reviews LLM consistency.",
            "sections": [
                {"title": "Core Idea", "body": "It studies whether models stay firm across follow-ups."},
                {"title": "Technical Contributions", "body": "It introduces PWC and CARG."},
            ],
        }
    }

    answer = answer_paper_question("what does this paper say", None, report)

    assert "Short answer" in answer["answer"]
    assert "stay firm" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_answers_document_wording_from_report() -> None:
    report = {
        "narrative_review": {
            "opening": "This document explains LLM reasoning engines.",
            "sections": [
                {"title": "Core Idea", "body": "It covers tokens, attention, prompting, and reasoning techniques."},
                {"title": "Key Claims", "body": "Prompting can improve task performance without fine-tuning."},
            ],
        }
    }

    answer = answer_paper_question("what does this document say?", None, report)

    assert "Short answer" in answer["answer"]
    assert "LLM reasoning engines" in answer["answer"]
    assert "not have enough evidence" not in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_answers_tell_me_about_this_from_report() -> None:
    report = {
        "narrative_review": {
            "opening": "This paper evaluates multi-turn reliability.",
            "sections": [
                {"title": "Core Idea", "body": "It asks whether LLMs remain consistent."},
                {"title": "Empirical Findings", "body": "Follow-up pressure can sway models."},
            ],
        }
    }

    answer = answer_paper_question("so tell me about this", None, report)

    assert "Short answer" in answer["answer"]
    assert "multi-turn reliability" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_gives_reading_guide() -> None:
    report = {
        "narrative_review": {
            "sections": [
                {"title": "Core Idea", "body": "Reliability is multi-turn stability."},
                {"title": "Skeptical Notes", "body": "Check the statistics carefully."},
            ],
        }
    }

    answer = answer_paper_question("how should i go through it", None, report)

    assert "Go through it in this order" in answer["answer"]
    assert "specific follow-up questions" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_gives_reading_guide_for_study_wording() -> None:
    report = {
        "narrative_review": {
            "sections": [
                {"title": "Core Idea", "body": "Start with how LLMs process tokens and attention."},
                {"title": "Key Claims", "body": "Then study prompting and reasoning patterns."},
                {"title": "Skeptical Notes", "body": "Check which claims have citations."},
            ],
        }
    }

    answer = answer_paper_question("how should I study this?", None, report)

    assert "Go through it in this order" in answer["answer"]
    assert "tokens and attention" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_uses_key_claims_for_contribution_question() -> None:
    report = {
        "narrative_review": {
            "opening": "This document explains LLM reasoning.",
            "sections": [
                {"title": "Core Idea", "body": "It explains how language models reason."},
                {"title": "Key Claims", "body": "Prompt engineering can improve accuracy and control format."},
            ],
        }
    }

    answer = answer_paper_question("what are the main contributions?", None, report)

    assert "Main contributions or key points" in answer["answer"]
    assert "Prompt engineering" in answer["answer"]
    assert "not have enough evidence" not in answer["answer"]


def test_paper_chat_answers_weakness_question_from_skeptical_notes() -> None:
    report = {
        "narrative_review": {
            "sections": [
                {"title": "Core Idea", "body": "It explains LLM reasoning."},
                {"title": "Skeptical Notes", "body": "No bracket-style citations were detected."},
            ],
        }
    }

    answer = answer_paper_question("what are the weaknesses?", None, report)

    assert "No bracket-style citations" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_summarizes_llm_module_as_topics() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines SECTION 1 Fundamentals of Large Language Models.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": (
                        "Token Embeddings and Positional Encoding. Self-Attention Mechanism uses Query, Key, and Value. "
                        "Context window includes system prompt, conversation history, retrieved memory, and tool definitions. "
                        "Prompt engineering includes zero-shot prompting and anatomy of an effective prompt. "
                        "Chain-of-thought and reflection prompting improve reasoning. Hallucination risks need RAG and citation verification."
                    ),
                },
                {"title": "Key Claims", "body": "Prompt engineering can improve accuracy and control format."},
            ],
        }
    }

    answer = answer_paper_question("what does this document say?", None, report)

    assert "teaching module" in answer["answer"]
    assert "Main topics" in answer["answer"]
    assert "self-attention" in answer["answer"].lower()
    assert "MODULE 4" not in answer["answer"]


def test_paper_chat_gives_module_specific_study_order() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines SECTION 1 Fundamentals of Large Language Models.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": (
                        "Token Embeddings, Positional Encoding, Self-Attention, Context Window, "
                        "Prompt Engineering, Chain-of-Thought, Reflection Prompting, Hallucinations, RAG, and citations."
                    ),
                }
            ],
        }
    }

    answer = answer_paper_question("how should I study this?", None, report)

    assert "tokens, embeddings" in answer["answer"]
    assert "context windows" in answer["answer"]
    assert "verification" in answer["answer"]


def test_paper_chat_answers_llm_module_concept_question() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines SECTION 1 Fundamentals of Large Language Models.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Positional Encoding, Self-Attention, Context Window, Prompt Engineering, Chain-of-Thought, RAG.",
                }
            ],
        }
    }

    answer = answer_paper_question("explain self-attention", None, report)

    assert "Query, Key, and Value" in answer["answer"]
    assert "not have enough evidence" not in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_answers_llm_module_context_question() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines SECTION 1 Fundamentals of Large Language Models.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Self-Attention, Context Window, Prompt Engineering, Chain-of-Thought.",
                }
            ],
        }
    }

    answer = answer_paper_question("what is a context window?", None, report)

    assert "instructions" in answer["answer"]
    assert "conversation history" in answer["answer"]
    assert answer["mode"] == "report"


def test_paper_chat_uses_discussion_memory_for_known_entity() -> None:
    report = {
        "narrative_review": {
            "sections": [
                {
                    "title": "Technical Contributions",
                    "body": (
                        "Position-Weighted Consistency (PWC): a stability metric.\n\n"
                        "Confidence-Aware Response Generation (CARG): a mitigation method."
                    ),
                }
            ],
        }
    }
    memory = initialize_discussion_memory(report)
    first_answer = answer_paper_question("what are the technical contributions", None, report)
    memory = update_discussion_memory(memory, "what are the technical contributions", first_answer, report)

    answer = answer_paper_question("what is PWC?", None, report, discussion_memory=memory)

    assert "Earlier in this discussion" in answer["answer"]
    assert "stability metric" in answer["answer"]
    assert "Discussion memory" in answer["sources"]


def test_paper_chat_guards_when_evidence_is_too_weak() -> None:
    answer = answer_paper_question("what is transfer learning?", None)

    assert "not have enough evidence" in answer["answer"]
    assert "Enable web search" in answer["answer"]
    assert answer["mode"] == "guarded"
    assert answer["confidence"] == "Low"


def test_paper_chat_uses_web_only_when_enabled() -> None:
    captured_queries = []

    def fake_web_search(_question: str) -> list[dict]:
        captured_queries.append(_question)
        return [
            {
                "source": "Wikipedia",
                "title": "Transfer learning",
                "snippet": "Transfer learning reuses knowledge from one task on another task.",
                "url": "https://example.com/transfer-learning",
            }
        ]

    answer = answer_paper_question(
        "what is transfer learning?",
        None,
        web_search_enabled=True,
        web_search_fn=fake_web_search,
    )

    assert "external context" in answer["answer"]
    assert answer["mode"] == "web"
    assert "Web" in answer["sources"]
    assert captured_queries == ["transfer learning"]


def test_paper_chat_reports_web_search_failure() -> None:
    def fake_web_search(_question: str) -> list[dict]:
        return [{"source": "Wikipedia", "query": "transfer learning", "error": "network unavailable"}]

    answer = answer_paper_question(
        "what is transfer learning?",
        None,
        web_search_enabled=True,
        web_search_fn=fake_web_search,
    )

    assert "Web search was enabled" in answer["answer"]
    assert answer["mode"] == "web_error"
    assert answer["web_errors"] == ["network unavailable"]
    assert answer["confidence"] == "Low"


def test_skeptic_check_claim_returns_supported_verdict() -> None:
    text = "Confidence-Aware Response Generation improves response stability in multi-turn settings."
    store = build_paper_chat_store(text, "demo.pdf")

    answer = skeptic_check_claim("CARG improves response stability.", store)

    assert answer["verdict"] in {"Supported", "Partly supported"}
    assert "Best paper evidence" in answer["answer"]
    assert answer["evidence"]
    assert answer["mode"] == "skeptic"


def test_skeptic_check_claim_flags_unsupported_claim() -> None:
    text = "The paper studies response stability in multi-turn settings."
    store = build_paper_chat_store(text, "demo.pdf")

    answer = skeptic_check_claim("The method guarantees perfect translation quality.", store)

    assert answer["verdict"] in {"Unsupported", "Needs more evidence"}
    assert answer["confidence"] == "Low"
    assert "direct paper evidence" in answer["follow_up_questions"][0]


def test_skeptic_check_claim_uses_report_when_vector_match_is_weak() -> None:
    text = "This unrelated passage is about tokenization and model families."
    store = build_paper_chat_store(text, "demo.pdf")
    report = {
        "narrative_review": {
            "sections": [
                {
                    "title": "Key Claims",
                    "body": "Prompt engineering can improve accuracy, reduce hallucination, and control output format.",
                }
            ]
        }
    }

    answer = skeptic_check_claim(
        "Prompt engineering can improve accuracy and control output format.",
        store,
        report=report,
    )

    assert answer["verdict"] == "Partly supported"
    assert answer["confidence"] == "Medium"
    assert "Final review" in answer["sources"]
    assert "Prompt engineering" in answer["evidence"][0]["text"]


def test_skeptic_check_claim_uses_llm_module_evidence_for_hallucination_claim() -> None:
    report = {
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Self-Attention, Prompt Engineering, Chain-of-Thought, Hallucinations, RAG, citations, and verification.",
                }
            ],
        }
    }

    answer = skeptic_check_claim(
        "LLMs can hallucinate and need verification.",
        None,
        report=report,
    )

    assert answer["verdict"] == "Partly supported"
    assert "hallucinations" in answer["evidence"][0]["text"]
    assert "Final review" in answer["sources"]


def test_skeptic_check_claim_flags_contradiction() -> None:
    text = "The method does not improve response stability on the hardest evaluation subset."
    store = build_paper_chat_store(text, "demo.pdf")

    answer = skeptic_check_claim("The method improves response stability.", store)

    assert answer["verdict"] == "Contradicted"
    assert "opposite" in answer["confidence_reason"]
