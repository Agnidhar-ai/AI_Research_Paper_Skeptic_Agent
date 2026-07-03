import re

from tools.retriever import Retriever
from tools.vector_store import InMemoryVectorStore
from tools.text_cleaner import clean_display_text, clean_snippet_text


class EvidenceFinder:
    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self.retriever = Retriever(vector_store)

    def find_for_claims(self, claims: list[dict], top_k: int = 3) -> list[dict]:
        evidence = []
        for claim in claims:
            claim_text = claim["claim"]
            matches = self.retriever.search(claim_text, top_k=top_k)
            evidence.append(
                {
                    "claim": claim_text,
                    "risk_level": claim["risk_level"],
                    "matches": [
                        {
                            "score": match["score"],
                            "text": self._best_snippet(match["text"], claim_text),
                            "metadata": match["metadata"],
                        }
                        for match in matches
                    ],
                }
            )
        return evidence

    def _best_snippet(self, text: str, claim_text: str) -> str:
        clean_text = clean_display_text(text)
        candidates = self._candidate_snippets(clean_text, claim_text)
        if candidates:
            return clean_snippet_text(candidates[0], limit=420)
        return clean_snippet_text(clean_text)

    def _candidate_snippets(self, text: str, claim_text: str) -> list[str]:
        claim_terms = self._important_terms(claim_text)
        pieces = [
            piece.strip()
            for piece in re.split(
                r"(?<=[.!?])\s+|(?=\b(?:Abstract|Introduction|First|Second|Third|Experimental results|Current research|To address|Conversely|Given|As shown|Collectively)\b)",
                text,
            )
            if len(piece.strip()) > 45
        ]

        windows = []
        for term in sorted(claim_terms, key=len, reverse=True):
            match = re.search(re.escape(term), text, flags=re.IGNORECASE)
            if not match:
                continue
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 260)
            start = text.rfind(" ", 0, start) + 1 if start else 0
            end_boundary = text.find(" ", end)
            if end_boundary != -1:
                end = end_boundary
            windows.append(text[start:end].strip())

        candidates = pieces + windows
        candidates = [candidate for candidate in candidates if candidate]
        return sorted(
            candidates,
            key=lambda candidate: self._snippet_score(candidate, claim_terms),
            reverse=True,
        )

    def _snippet_score(self, snippet: str, claim_terms: set[str]) -> float:
        terms = self._important_terms(snippet)
        score = len(claim_terms.intersection(terms)) * 5
        lower_snippet = snippet.lower()
        if "anonymous author" in lower_snippet or "affiliation address email" in lower_snippet:
            score -= 8
        if lower_snippet.startswith("firm or fickle"):
            score -= 6
        if snippet and snippet[0].islower():
            score -= 2
        if 80 <= len(snippet) <= 420:
            score += 2
        return score

    def _important_terms(self, text: str) -> set[str]:
        stop_words = {
            "about",
            "across",
            "after",
            "against",
            "being",
            "could",
            "demonstrate",
            "demonstrates",
            "during",
            "especially",
            "framework",
            "improve",
            "improves",
            "including",
            "model",
            "models",
            "paper",
            "response",
            "results",
            "should",
            "significant",
            "significantly",
            "their",
            "these",
            "third",
            "using",
            "where",
            "which",
            "while",
            "with",
        }
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
            if token.lower() not in stop_words
        }
