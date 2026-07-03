from tools import source_verifier
from tools.source_verifier import (
    build_section_support_query,
    build_verification_queries,
    search_arxiv,
    search_external_support_for_section,
    search_wikipedia,
    summarize_external_verification,
    verify_external_sources,
)


def test_build_verification_queries_finds_llm_consistency_terms() -> None:
    report = {
        "summary": "This paper introduces Position-Weighted Consistency and Confidence-Aware Response Generation for Large Language Models.",
        "claims": [{"claim": "CARG improves consistency in multi-turn interactions."}],
    }

    queries = build_verification_queries(report)

    assert "large language model consistency multi-turn interaction" in queries


def test_build_verification_queries_prefers_llm_module_terms() -> None:
    report = {
        "summary": "MODULE 4 LLMs as Reasoning Engines.",
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Positional Encoding, Self-Attention, Prompt Engineering, and Chain-of-Thought.",
                }
            ],
        },
        "claims": [{"claim": "Prompt engineering can improve accuracy."}],
    }

    queries = build_verification_queries(report)

    assert queries[0] == "large language models reasoning engines prompt engineering"
    assert "large language model consistency multi-turn interaction" not in queries


def test_search_wikipedia_uses_injected_fetcher() -> None:
    def fake_fetch(_url: str) -> dict:
        return {"query": {"search": [{"title": "Large language model", "pageid": 1, "snippet": "A <b>model</b>."}]}}

    results = search_wikipedia("large language model", fetch_json_fn=fake_fetch)

    assert results[0]["title"] == "Large language model"
    assert results[0]["snippet"] == "A model."


def test_search_arxiv_uses_injected_fetcher() -> None:
    def fake_fetch(_url: str) -> str:
        return """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/1234.5678</id>
            <title>Consistency in LLMs</title>
            <summary>Study of consistency.</summary>
          </entry>
        </feed>"""

    results = search_arxiv("consistency LLM", fetch_text_fn=fake_fetch)

    assert results[0]["title"] == "Consistency in LLMs"


def test_build_section_support_query_prefers_known_llm_terms() -> None:
    section = {
        "title": "Technical Contributions",
        "content": "Position-Weighted Consistency (PWC) measures whether large language models stay stable.",
    }

    query = build_section_support_query(section)

    assert query == "position-weighted consistency large language models"


def test_build_section_support_query_handles_llm_module_overview() -> None:
    section = {
        "title": "Overview",
        "content": "MODULE 4 LLMs as Reasoning Engines covers Token Embeddings, Positional Encoding, and Self-Attention.",
    }

    query = build_section_support_query(section)

    assert query == "transformer self-attention token embeddings large language models"


def test_search_external_support_for_section_uses_injected_searchers() -> None:
    def fake_wikipedia(query: str, limit: int = 2) -> list[dict]:
        return [
            {
                "source": "Wikipedia",
                "query": query,
                "title": "Large language model",
                "snippet": "Large language models can be evaluated for behavior.",
                "url": "https://example.com/wiki",
            }
        ]

    def fake_arxiv(query: str, limit: int = 2) -> list[dict]:
        return [
            {
                "source": "arXiv",
                "query": query,
                "title": "Consistency in LLMs",
                "snippet": "Study of consistency in multi-turn interactions.",
                "url": "https://example.com/arxiv",
            }
        ]

    support = search_external_support_for_section(
        {"title": "Core Idea", "content": "The paper studies multi-turn LLM consistency."},
        wikipedia_search_fn=fake_wikipedia,
        arxiv_search_fn=fake_arxiv,
    )

    assert support["query"] == "large language model consistency multi-turn interactions"
    assert support["status"] == "ok"
    assert support["result_count"] == 2
    assert support["wikipedia"][0]["title"] == "Large language model"
    assert support["arxiv"][0]["title"] == "Consistency in LLMs"


def test_verify_external_sources_reports_partial_success(monkeypatch) -> None:
    def fake_wikipedia(query: str, limit: int = 3) -> list[dict]:
        return [
            {
                "source": "Wikipedia",
                "query": query,
                "title": "Large language model",
                "snippet": "Transformer-based model.",
                "url": "https://example.com/wiki",
            }
        ]

    def fake_arxiv(query: str, limit: int = 3) -> list[dict]:
        return [{"source": "arXiv", "query": query, "error": "The read operation timed out"}]

    monkeypatch.setattr(source_verifier, "search_wikipedia", fake_wikipedia)
    monkeypatch.setattr(source_verifier, "search_arxiv", fake_arxiv)

    report = {
        "summary": "MODULE 4 LLMs as Reasoning Engines.",
        "narrative_review": {
            "opening": "MODULE 4 LLMs as Reasoning Engines.",
            "sections": [
                {
                    "title": "Core Idea",
                    "body": "Token Embeddings, Positional Encoding, Self-Attention, and Prompt Engineering.",
                }
            ],
        },
    }

    result = verify_external_sources(report, max_queries=1)

    assert result["status"] == "partial"
    assert result["result_count"] == 1
    assert result["errors"] == ["arXiv: The read operation timed out"]
    assert result["status_summary"]["label"] == "Partial external check"


def test_external_support_reports_unavailable_when_all_sources_fail() -> None:
    def fake_wikipedia(query: str, limit: int = 2) -> list[dict]:
        return [{"source": "Wikipedia", "query": query, "error": "network unavailable"}]

    def fake_arxiv(query: str, limit: int = 2) -> list[dict]:
        return [{"source": "arXiv", "query": query, "error": "network unavailable"}]

    support = search_external_support_for_section(
        {"title": "Core Idea", "content": "The paper studies multi-turn LLM consistency."},
        wikipedia_search_fn=fake_wikipedia,
        arxiv_search_fn=fake_arxiv,
    )

    assert support["status"] == "unavailable"
    assert support["result_count"] == 0
    assert support["status_summary"]["label"] == "External sources unavailable"


def test_external_summary_warns_that_sources_are_context_not_validation() -> None:
    summary = summarize_external_verification("ok", 2, [])

    assert summary["tone"] == "success"
    assert "do not validate the paper's experiment" in summary["message"]
