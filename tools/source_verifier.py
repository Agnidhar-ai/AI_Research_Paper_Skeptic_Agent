from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from tools.text_cleaner import clean_display_text


def verify_external_sources(report: dict, max_queries: int = 3, results_per_source: int = 3) -> dict:
    queries = build_verification_queries(report, limit=max_queries)
    wikipedia_results = [
        result
        for query in queries
        for result in search_wikipedia(query, limit=results_per_source)
    ]
    arxiv_results = [
        result
        for query in queries
        for result in search_arxiv(query, limit=results_per_source)
    ]
    all_results = [*wikipedia_results, *arxiv_results]
    valid_results = [result for result in all_results if not result.get("error")]
    errors = [
        f"{result.get('source', 'Source')}: {result.get('error')}"
        for result in all_results
        if result.get("error")
    ]
    status = _external_status(valid_results, errors)
    return {
        "disclaimer": (
            "External sources provide background context only. They do not prove the paper's claims; "
            "important claims still need source-by-source verification."
        ),
        "queries": queries,
        "wikipedia": wikipedia_results,
        "arxiv": arxiv_results,
        "result_count": len(valid_results),
        "errors": errors,
        "status": status,
        "status_summary": summarize_external_verification(status, len(valid_results), errors),
    }


def search_external_support_for_section(
    section: dict,
    results_per_source: int = 2,
    wikipedia_search_fn=None,
    arxiv_search_fn=None,
) -> dict:
    query = build_section_support_query(section)
    wikipedia_search_fn = wikipedia_search_fn or search_wikipedia
    arxiv_search_fn = arxiv_search_fn or search_arxiv

    wikipedia_results = wikipedia_search_fn(query, limit=results_per_source)
    arxiv_results = arxiv_search_fn(query, limit=results_per_source)
    errors = [
        f"{result.get('source', 'Source')}: {result.get('error')}"
        for result in [*wikipedia_results, *arxiv_results]
        if result.get("error")
    ]
    valid_results = [
        result
        for result in [*wikipedia_results, *arxiv_results]
        if not result.get("error")
    ]
    status = _external_status(valid_results, errors)

    return {
        "disclaimer": (
            "External support is background context only. It can support or clarify parts of a claim, "
            "but it does not prove that this paper's specific experiment is correct."
        ),
        "query": query,
        "status": status,
        "result_count": len(valid_results),
        "wikipedia": wikipedia_results,
        "arxiv": arxiv_results,
        "errors": errors,
        "status_summary": summarize_external_verification(status, len(valid_results), errors),
    }


def summarize_external_verification(status: str, result_count: int, errors: list[str] | None = None) -> dict:
    errors = errors or []
    if status == "ok":
        return {
            "label": "Background sources found",
            "message": (
                f"Found {result_count} external background source(s). Use them for context only; "
                "they do not validate the paper's experiment."
            ),
            "tone": "success",
        }
    if status == "partial":
        first_error = errors[0] if errors else "one source could not be checked"
        return {
            "label": "Partial external check",
            "message": (
                f"Found {result_count} external background source(s), but at least one source check failed "
                f"({first_error}). Treat this as incomplete context, not validation."
            ),
            "tone": "warning",
        }
    if status == "unavailable":
        first_error = errors[0] if errors else "external sources were unreachable"
        return {
            "label": "External sources unavailable",
            "message": (
                f"No usable external source results were returned because checks failed ({first_error}). "
                "The review still uses the uploaded document, but web/source context is unavailable."
            ),
            "tone": "error",
        }
    return {
        "label": "No usable external results",
        "message": (
            "The external search ran but did not return usable background sources. Try a more specific section "
            "or verify the claim manually."
        ),
        "tone": "warning",
    }


def build_section_support_query(section: dict) -> str:
    title = clean_display_text(section.get("title", ""))
    content = clean_display_text(section.get("content", ""))
    lower_text = f"{title} {content}".lower()

    if "position-weighted consistency" in lower_text or "pwc" in lower_text:
        return "position-weighted consistency large language models"
    if "confidence-aware response generation" in lower_text or "carg" in lower_text:
        return "confidence-aware response generation large language models"
    if _looks_like_llm_reasoning_module(lower_text):
        if "self-attention" in lower_text or "token" in lower_text or "embedding" in lower_text:
            return "transformer self-attention token embeddings large language models"
        if "prompt" in lower_text or "chain-of-thought" in lower_text:
            return "prompt engineering chain-of-thought large language models"
        return "large language models reasoning engines prompt engineering"
    if "mt-consistency" in lower_text or "multi-turn" in lower_text:
        return "large language model consistency multi-turn interactions"
    if "citation" in lower_text or "reference" in lower_text:
        return "research paper citation verification"

    key_phrase = _first_informative_phrase(content)
    return key_phrase or title or "research paper evidence"


def build_verification_queries(report: dict, limit: int = 3) -> list[str]:
    text_parts = [report.get("summary", "")]
    narrative = report.get("narrative_review") or {}
    text_parts.append(narrative.get("opening", ""))
    text_parts.extend(section.get("body", "") for section in narrative.get("sections", []))
    text_parts.extend(
        claim.get("claim", "")
        for claim in report.get("claims", [])
        if isinstance(claim, dict)
    )
    text = clean_display_text(" ".join(text_parts))
    lower_text = text.lower()

    preferred_queries = []
    if _looks_like_llm_reasoning_module(lower_text):
        preferred_queries.extend(
            [
                "large language models reasoning engines prompt engineering",
                "transformer self-attention token embeddings",
                "chain-of-thought prompting large language models",
            ]
        )
        return preferred_queries[:limit]

    mentions_llms = "large language model" in lower_text or "llm" in lower_text
    if mentions_llms and ("consistency" in lower_text or "multi-turn" in lower_text):
        preferred_queries.append("large language model consistency multi-turn interaction")
    if "confidence-aware response generation" in lower_text or "carg" in lower_text:
        preferred_queries.append("confidence-aware response generation language model")
    if "position-weighted consistency" in lower_text or "pwc" in lower_text:
        preferred_queries.append("position-weighted consistency language models")
    if "mt-consistency" in lower_text:
        preferred_queries.append("MT-Consistency benchmark language models")

    phrase_queries = _extract_key_phrases(text)
    queries = []
    for query in preferred_queries + phrase_queries:
        if query and query.lower() not in {item.lower() for item in queries}:
            queries.append(query)
        if len(queries) >= limit:
            break

    return queries or ["research paper methodology evaluation"]


def search_wikipedia(query: str, limit: int = 3, fetch_json_fn=None) -> list[dict]:
    fetch_json_fn = fetch_json_fn or _fetch_json
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
            "utf8": 1,
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    try:
        payload = fetch_json_fn(url)
    except Exception as exc:
        return [{"source": "Wikipedia", "query": query, "error": str(exc)}]

    results = []
    for item in payload.get("query", {}).get("search", []):
        title = clean_display_text(item.get("title", ""))
        page_id = item.get("pageid")
        results.append(
            {
                "source": "Wikipedia",
                "query": query,
                "title": title,
                "snippet": clean_display_text(_strip_html(item.get("snippet", ""))),
                "url": f"https://en.wikipedia.org/?curid={page_id}" if page_id else "",
            }
        )
    return results


def search_arxiv(query: str, limit: int = 3, fetch_text_fn=None) -> list[dict]:
    fetch_text_fn = fetch_text_fn or _fetch_text
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"

    try:
        payload = fetch_text_fn(url)
    except Exception as exc:
        return [{"source": "arXiv", "query": query, "error": str(exc)}]

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        return [{"source": "arXiv", "query": query, "error": f"Could not read arXiv response: {exc}"}]
    results = []
    for entry in root.findall("atom:entry", namespace):
        title = clean_display_text(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = clean_display_text(entry.findtext("atom:summary", default="", namespaces=namespace))
        link = entry.findtext("atom:id", default="", namespaces=namespace)
        results.append(
            {
                "source": "arXiv",
                "query": query,
                "title": title,
                "snippet": summary[:500],
                "url": link,
            }
        )
    return results


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "AIResearchPaperSkepticAgent/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "AIResearchPaperSkepticAgent/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8")


def _external_status(valid_results: list[dict], errors: list[str]) -> str:
    if valid_results and errors:
        return "partial"
    if valid_results:
        return "ok"
    if errors:
        return "unavailable"
    return "no_results"


def _extract_key_phrases(text: str) -> list[str]:
    acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", text)
    title_phrases = re.findall(r"\b[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){1,4}\b", text)
    phrases = [phrase for phrase in title_phrases if len(phrase.split()) <= 5]
    return [*phrases, *acronyms]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _looks_like_llm_reasoning_module(lower_text: str) -> bool:
    module_marker = "llms as reasoning engines" in lower_text or ("module" in lower_text and "large language models" in lower_text)
    mechanics_marker = any(
        marker in lower_text
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


def _first_informative_phrase(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", sentences[0])
    stopwords = {
        "the",
        "this",
        "that",
        "paper",
        "section",
        "report",
        "analyzes",
        "proposes",
        "shows",
        "uses",
    }
    useful_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(useful_words[:8])
