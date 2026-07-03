import re

from tools.text_cleaner import clean_display_text


class CitationAgent:
    def extract_citation_checks(self, paper_text: str) -> list[str]:
        citations = self.extract_citation_details(paper_text)

        if not citations:
            return ["No bracket-style citations were detected in the extracted text."]

        return [
            f"{item['citation']}: {item['check']}"
            for item in citations[:20]
        ]

    def extract_citation_details(self, paper_text: str) -> list[dict]:
        sentences = self._split_sentences(paper_text)
        details = []
        seen = set()

        for sentence in sentences:
            for citation in re.findall(r"\[[0-9,\-\s]+\]", sentence):
                if citation in seen:
                    continue
                seen.add(citation)
                details.append(
                    {
                        "citation": citation,
                        "count": len(self._citation_numbers(citation)),
                        "context": clean_display_text(sentence),
                        "check": self._check_label(citation, sentence),
                    }
                )

        return details

    def _split_sentences(self, text: str) -> list[str]:
        compact_text = clean_display_text(text)
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact_text)
            if sentence.strip()
        ]

    def _citation_numbers(self, citation: str) -> list[int]:
        numbers = []
        for part in citation.strip("[]").split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                if start.strip().isdigit() and end.strip().isdigit():
                    numbers.extend(range(int(start), int(end) + 1))
            elif part.isdigit():
                numbers.append(int(part))
        return numbers

    def _check_label(self, citation: str, sentence: str) -> str:
        count = len(self._citation_numbers(citation))
        lower_sentence = sentence.lower()
        if count >= 3:
            return "Grouped citation; verify each referenced work supports the broad background claim."
        if any(term in lower_sentence for term in ("demonstrate", "show", "significant", "improve", "superior")):
            return "Claim-bearing citation; verify it directly supports the empirical statement."
        return "Background citation; verify it supports the surrounding context."
