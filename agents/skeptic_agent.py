from agents.citation_agent import CitationAgent
from agents.claim_analyzer import ClaimAnalyzer
from agents.evidence_finder import EvidenceFinder
from agents.reasoning_layer import ReasoningLayer
from agents.reviewer_agent import ReviewerAgent
from config import settings
from tools.chunker import chunk_text
from tools.vector_store_factory import create_vector_store


class SkepticAgent:
    def __init__(self) -> None:
        self.reviewer = ReviewerAgent()
        self.citation_agent = CitationAgent()
        self.claim_analyzer = ClaimAnalyzer()
        self.reasoning_layer = ReasoningLayer()

    def review(self, paper_text: str, source_name: str = "paper") -> dict:
        chunks = chunk_text(paper_text, settings.chunk_size, settings.chunk_overlap)
        vector_store = create_vector_store()
        vector_store.add_texts(
            chunks,
            metadata=[{"source": source_name, "chunk": index} for index in range(len(chunks))],
        )

        review = self.reviewer.review(paper_text, source_name=source_name)
        citation_details = self.citation_agent.extract_citation_details(paper_text)
        citations = [
            f"{item['citation']}: {item['check']}"
            for item in citation_details[:20]
        ] or ["No bracket-style citations were detected in the extracted text."]
        claims = self.claim_analyzer.extract_claims(paper_text)
        evidence = EvidenceFinder(vector_store).find_for_claims(claims)
        reasoning = self.reasoning_layer.evaluate(
            claims=claims,
            evidence=evidence,
            citation_checks=citations,
            reviewer_findings=review["findings"],
        )

        return {
            "source": source_name,
            "pipeline": [
                "document_loader",
                "text_chunking",
                "knowledge_base",
                "skeptic_agent",
                "evidence_finder",
                "claim_analyzer",
                "reasoning_layer",
                "structured_report",
            ],
            "summary": review["summary"],
            "claims": claims,
            "evidence": evidence,
            "reasoning": reasoning,
            "skeptical_findings": reasoning["findings"],
            "citation_checks": citations,
            "citation_details": citation_details,
        }
