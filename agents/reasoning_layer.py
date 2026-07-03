class ReasoningLayer:
    def evaluate(
        self,
        claims: list[dict],
        evidence: list[dict],
        citation_checks: list[str],
        reviewer_findings: list[str],
    ) -> dict:
        findings = []

        for item in evidence:
            best_score = item["matches"][0]["score"] if item["matches"] else 0.0
            if item["risk_level"] == "high":
                findings.append(
                    {
                        "type": "strong_claim",
                        "severity": "high",
                        "claim": item["claim"],
                        "finding": "The claim uses strong language and should be checked against experimental evidence, baselines, and citations.",
                        "best_evidence_score": best_score,
                    }
                )
            elif best_score < 0.05:
                findings.append(
                    {
                        "type": "weak_evidence",
                        "severity": "medium",
                        "claim": item["claim"],
                        "finding": "The knowledge base did not surface strong supporting context for this claim.",
                        "best_evidence_score": best_score,
                    }
                )

        for finding in reviewer_findings:
            findings.append(
                {
                    "type": "reviewer_check",
                    "severity": "medium",
                    "finding": finding,
                }
            )

        if citation_checks and "No bracket-style citations" in citation_checks[0]:
            findings.append(
                {
                    "type": "missing_citation_pattern",
                    "severity": "medium",
                    "finding": citation_checks[0],
                }
            )

        return {
            "assumptions_checked": [
                "Claims should be supported by nearby evidence or retrievable context.",
                "Strong conclusions require clear baselines, dataset details, and limitations.",
                "Citation markers should be present where evidence from prior work is invoked.",
            ],
            "findings": findings or [
                {
                    "type": "no_major_issue_detected",
                    "severity": "low",
                    "finding": "No major heuristic concerns were detected in this pass.",
                }
            ],
            "claim_count": len(claims),
        }
