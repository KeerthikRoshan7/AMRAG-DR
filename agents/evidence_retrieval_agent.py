"""
Agent 2: Evidence Retrieval Agent (AM-RAG Section 5.3.2)

Responsibilities: Query vector database, retrieve ophthalmology guidelines,
retrieve treatment protocols, retrieve peer-reviewed evidence.
Knowledge Sources: AAO Guidelines, ICO Guidelines, DR Research Articles,
Hospital Protocols (represented locally by knowledge_base/documents/*.md).

Input: Structured lesion findings from Agent 1.
Output: Set of Top-K retrieved clinical evidence chunks R = {r1, ..., rk}
        (Section 5.3.3).
"""

from knowledge_base.retriever import KnowledgeRetriever


class EvidenceRetrievalAgent:
    def __init__(self, retriever: KnowledgeRetriever | None = None):
        self.retriever = retriever or KnowledgeRetriever()

    def build_clinical_query(self, lesion_findings: dict) -> str:
        """Turn structured lesion findings into a natural-language query
        for semantic retrieval (Section 5.4.2: Detected Lesions -> Clinical
        Query Generation)."""
        burden = lesion_findings["lesion_burden"]
        active_findings = [
            name.replace("_", " ") for name, val in burden.items() if val > 0.15
        ]
        findings_str = ", ".join(active_findings) if active_findings else "no significant lesions"

        query = (
            f"Diabetic retinopathy severity grading and management for a patient "
            f"presenting with: {findings_str}. Predicted severity level: "
            f"{lesion_findings['severity_label']}. What are the grading criteria, "
            f"follow-up interval, and treatment considerations?"
        )
        return query

    def retrieve(self, lesion_findings: dict, top_k: int = 5) -> list[dict]:
        query = self.build_clinical_query(lesion_findings)
        evidence = self.retriever.retrieve(query, top_k=top_k)
        return evidence
