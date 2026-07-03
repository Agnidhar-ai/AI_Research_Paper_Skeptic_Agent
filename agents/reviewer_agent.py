import re

from prompts.reviewer_prompt import REVIEWER_PROMPT
from config import settings
from tools.llm_client import LLMClient
from tools.text_cleaner import clean_display_text


class ReviewerAgent:
    def review(self, paper_text: str, source_name: str = "") -> dict:
        excerpt = self._build_summary(paper_text)
        lower_text = f"{source_name}\n{paper_text}".lower()
        findings = []

        if self._looks_like_learning_material(lower_text):
            findings.extend(
                [
                    "This appears to be instructional or slide-deck material rather than a full empirical research paper.",
                    "The review should focus on covered concepts, definitions, workflows, and missing references instead of expecting a standard experiment section.",
                    "Any tool or framework recommendation should still be checked against current documentation before use.",
                ]
            )
        else:
            if "limitation" not in lower_text:
                findings.append("The paper may understate limitations or threats to validity.")
            if "baseline" not in lower_text:
                findings.append("The evaluation may need clearer baseline comparisons.")
            if "dataset" not in lower_text:
                findings.append("Dataset details may be insufficient for reproducibility.")
            if "anonymous author" in lower_text:
                findings.append("Author and affiliation details are anonymized, so provenance and accountability cannot be assessed from this copy.")
            if "significant" in lower_text and not self._mentions_any(
                lower_text,
                ("p-value", "p value", "confidence interval", "standard deviation", "statistical test"),
            ):
                findings.append("The paper uses significance language, but the extracted text does not clearly show statistical tests, p-values, confidence intervals, or variance reporting.")
            if self._mentions_any(lower_text, ("benchmark", "dataset")) and not self._mentions_any(
                lower_text,
                ("github", "repository", "available at", "license", "released", "appendix"),
            ):
                findings.append("The benchmark or dataset may need clearer release, license, or access details for reproducibility.")
            if self._mentions_any(lower_text, ("confidence-aware", "confidence scores", "model confidence")) and not self._mentions_any(
                lower_text,
                ("calibration", "calibrated", "expected calibration error", "ece"),
            ):
                findings.append("The confidence-based method should explain whether confidence scores are calibrated and comparable across models.")
            if self._mentions_any(lower_text, ("healthcare", "medical", "patient", "education")) and not self._mentions_any(
                lower_text,
                ("human evaluation", "expert evaluation", "ethics", "irb", "clinician"),
            ):
                findings.append("The paper invokes high-stakes domains, but the extracted text does not clearly show domain-expert or ethics validation.")
            if self._mentions_any(lower_text, ("improves", "outperform", "superior")) and not self._mentions_any(
                lower_text,
                ("ablation", "sensitivity", "error analysis", "failure case"),
            ):
                findings.append("Performance claims would be stronger with visible ablations, sensitivity analysis, or failure-case discussion.")

        if not findings:
            findings.append("No obvious heuristic concerns found in the initial pass.")

        llm_notes = ""
        if settings.llm_provider.lower() != "heuristic":
            try:
                llm_notes = LLMClient().complete(
                    f"{REVIEWER_PROMPT}\n\nPaper excerpt:\n{excerpt}\n\nReturn a concise skeptical review."
                )
            except Exception as exc:
                llm_notes = f"LLM review skipped: {exc}"

        return {
            "prompt": REVIEWER_PROMPT,
            "summary": excerpt or "No text extracted from the paper.",
            "findings": findings,
            "llm_notes": llm_notes,
        }

    def _mentions_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def _looks_like_learning_material(self, text: str) -> bool:
        learning_markers = (
            "module",
            "learning objective",
            "lecture",
            "slides",
            "course",
            "langchain",
            "langgraph",
            "crewai",
            "agentic frameworks",
            "multi-agent systems",
            "deep learning",
            "deeplearning",
        )
        research_markers = ("abstract", "experiment", "methodology", "results", "p-value", "dataset")
        return (
            sum(marker in text for marker in learning_markers) >= 1
            and sum(marker in text for marker in research_markers) <= 2
        )

    def _build_summary(self, paper_text: str, max_sentences: int = 6) -> str:
        text = clean_display_text(paper_text)
        abstract_match = re.search(
            r"\bAbstract\b(.+?)(?:\bIntroduction\b|\b1\s+Introduction\b)",
            text,
            flags=re.IGNORECASE,
        )
        source = abstract_match.group(1).strip() if abstract_match else text
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", source)
            if len(sentence.strip()) > 20
        ]
        if sentences:
            return " ".join(sentences[:max_sentences])
        return source[:1800].rsplit(" ", 1)[0].strip()
