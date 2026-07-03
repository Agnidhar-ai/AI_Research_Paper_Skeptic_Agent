import re

from tools.text_cleaner import clean_display_text, clean_extracted_text


CLAIM_SIGNALS = (
    "show",
    "shows",
    "demonstrate",
    "demonstrates",
    "prove",
    "proves",
    "outperform",
    "outperforms",
    "improve",
    "improves",
    "significant",
    "significantly",
    "novel",
    "state-of-the-art",
    "sota",
)

TOPIC_STOPWORDS = {
    "abstract",
    "agenda",
    "contents",
    "given",
    "how to",
    "how much is",
    "introduction",
    "line",
    "mate",
    "overview",
    "section",
    "summary",
    "thank",
    "thanks",
    "this",
    "write",
}


class ClaimAnalyzer:
    def extract_claims(self, paper_text: str, limit: int = 12) -> list[dict]:
        sentences = self._split_sentences(paper_text)
        claims = []

        for sentence in sentences:
            signals = [
                signal
                for signal in CLAIM_SIGNALS
                if self._contains_signal(sentence, signal)
            ]
            if signals:
                claims.append(
                    {
                        "claim": clean_display_text(sentence),
                        "risk_level": self._risk_level(sentence),
                        "signals": signals,
                    }
                )

            if len(claims) >= limit:
                break

        if claims:
            return claims

        topic_claims = self._extract_topic_claims(paper_text, limit=limit)
        if topic_claims:
            return topic_claims

        return [
            {
                "claim": "No explicit high-confidence claims were detected by the heuristic analyzer.",
                "risk_level": "low",
                "signals": [],
            }
        ]

    def _split_sentences(self, text: str) -> list[str]:
        compact_text = clean_display_text(text)
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact_text)
            if len(sentence.strip()) > 40
        ]

    def _risk_level(self, claim: str) -> str:
        high_risk_terms = ("prove", "guarantee", "state-of-the-art", "sota")
        medium_risk_terms = (
            "significant",
            "significantly",
            "outperform",
            "novel",
            "high-stakes",
            "critical",
            "healthcare",
            "medical",
        )

        if any(self._contains_signal(claim, term) for term in high_risk_terms):
            return "high"
        if any(self._contains_signal(claim, term) for term in medium_risk_terms):
            return "medium"
        return "low"

    def _contains_signal(self, sentence: str, signal: str) -> bool:
        escaped_signal = re.escape(signal).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-zA-Z0-9_]){escaped_signal}(?![a-zA-Z0-9_])", sentence, re.IGNORECASE) is not None

    def _extract_topic_claims(self, text: str, limit: int) -> list[dict]:
        extracted_text = clean_extracted_text(text)
        candidates = self._domain_topic_candidates(extracted_text)
        for line in extracted_text.splitlines():
            line = clean_display_text(line)
            if self._looks_like_topic_line(line):
                candidates.append(line)

        if not candidates:
            compact_text = clean_display_text(text)
            candidates = self._topic_candidates(compact_text)

        topics = []
        seen = set()
        for topic in candidates:
            normalized = topic.lower()
            if normalized in seen or normalized in TOPIC_STOPWORDS:
                continue
            if not self._looks_like_topic_line(topic):
                continue
            seen.add(normalized)
            topics.append(topic)
            if len(topics) >= limit:
                break

        return [
            {
                "claim": f"The document covers the topic: {topic}.",
                "risk_level": "low",
                "signals": ["topic"],
            }
            for topic in topics
        ]

    def _domain_topic_candidates(self, text: str) -> list[str]:
        lower_text = text.lower()
        topic_rules = [
            (("linear equation", "matrixform", "matrix form", "vector form"), "Linear algebra and matrix form"),
            (("neuron", "aneuron", "bias"), "Neurons and bias terms"),
            (("derivative", "rate of change"), "Derivatives and rate of change"),
            (("function", "mapping"), "Functions as input-output mappings"),
            (("loss", "objective function"), "Loss and objective functions"),
            (("optimization", "minimize", "maximize"), "Optimization"),
            (("gradient decent", "gradient descent"), "Gradient descent"),
            (("sgd", "adam", "adagrad"), "Optimizers: SGD, Adam, and AdaGrad"),
        ]
        topics = []
        for markers, topic in topic_rules:
            if any(marker in lower_text for marker in markers):
                topics.append(topic)
        return topics

    def _topic_candidates(self, text: str) -> list[str]:
        phrases = re.findall(
            r"\b[A-Z][A-Za-z0-9&/+-]*(?:\s+[A-Z][A-Za-z0-9&/+-]*){0,3}\b",
            text,
        )
        cleaned = []
        for phrase in phrases:
            phrase = clean_display_text(phrase).strip(" -:|")
            words = phrase.split()
            if not words:
                continue
            if len(words) == 1 and len(words[0]) < 4:
                continue
            cleaned.append(phrase)
        return cleaned

    def _looks_like_topic_line(self, text: str) -> bool:
        text = clean_display_text(text).strip(" -:|.")
        if len(text) < 4 or len(text) > 80:
            return False
        if not re.search(r"[A-Za-z]", text):
            return False
        if any(symbol in text for symbol in ("=", "∑", "√", "≤", "≥")):
            return False
        if "." in text:
            return False
        words = text.split()
        if len(words) > 9:
            return False
        first_word = re.sub(r"[^A-Za-z]", "", words[0]) if words else ""
        if first_word and not (first_word[0].isupper() or first_word.isupper()):
            return False
        short_words = [word for word in words if len(re.sub(r"[^A-Za-z]", "", word)) <= 2]
        if len(short_words) > max(1, len(words) // 2):
            return False
        alpha_chars = sum(1 for char in text if char.isalpha())
        if alpha_chars / max(len(text), 1) < 0.55:
            return False
        weird_tokens = [
            word
            for word in words
            if len(word) >= 8 and not re.search(r"[aeiouAEIOU]", word)
        ]
        if weird_tokens:
            return False
        return True
