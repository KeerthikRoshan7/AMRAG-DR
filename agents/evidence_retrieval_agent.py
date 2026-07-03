"""
Agent 2: Evidence Retrieval Agent

Responsibilities: Query local FAISS KB and live public sources (PubMed,
Europe PMC, ClinicalTrials.gov), merge and rank into a single evidence set.

Knowledge Sources:
  - Local: AAO/ICO Guidelines, Hospital Protocols (knowledge_base/documents/*.md)
  - Live: PubMed, Europe PMC, ClinicalTrials.gov (knowledge_base/web_retriever.py)

Input: Structured lesion findings from Agent 1.
Output: Set of Top-K retrieved clinical evidence chunks R = {r1, ..., rk},
each carrying a source_url and clean sentence-level text for citation display
(Section 5.3.3).
"""

from __future__ import annotations

import asyncio
import concurrent.futures

from knowledge_base.retriever import KnowledgeRetriever
from knowledge_base.web_retriever import WebEvidenceRetriever, EvidenceChunk


class EvidenceRetrievalAgent:
    def __init__(
        self,
        local_retriever: KnowledgeRetriever | None = None,
        web_retriever: WebEvidenceRetriever | None = None,
        use_web: bool = True,
    ):
        self.local_retriever = local_retriever or KnowledgeRetriever()
        self.web_retriever = web_retriever or WebEvidenceRetriever(
            embedder=getattr(self.local_retriever, "embedder", None)
        )
        self.use_web = use_web

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

    async def retrieve(self, lesion_findings: dict, top_k: int = 5) -> list[dict]:
        query = self.build_clinical_query(lesion_findings)

        # Local KB retrieval stays synchronous (FAISS is fast in-process),
        # web retrieval runs concurrently against it.
        local_task = asyncio.to_thread(self.local_retriever.retrieve, query, top_k)

        if self.use_web:
            web_task = self.web_retriever.retrieve(query, top_k=top_k)
            local_results, web_chunks = await asyncio.gather(
                local_task, web_task, return_exceptions=True
            )
        else:
            local_results = await local_task
            web_chunks = []

        merged: list[EvidenceChunk] = []

        if isinstance(local_results, Exception):
            print(f"[EvidenceRetrievalAgent] local retrieval failed: {local_results!r}")
        else:
            for r in local_results:
                merged.append(EvidenceChunk(
                    text=r["text"],
                    source_url=r.get("source", "local://knowledge_base/documents"),
                    title=r.get("source", "Local clinical protocol"),
                    source_type="local",
                    relevance_score=r.get("score", 0.0),
                ))

        if isinstance(web_chunks, Exception):
            print(f"[EvidenceRetrievalAgent] web retrieval failed: {web_chunks!r}")
        else:
            merged.extend(web_chunks)

        # Global re-rank across local + web, then dedupe near-identical sentences
        merged.sort(key=lambda c: c.relevance_score, reverse=True)
        deduped = self._dedupe(merged)

        return [c.to_dict() for c in deduped[:top_k]]

    def retrieve_sync(self, lesion_findings: dict, top_k: int = 5) -> list[dict]:
        """Sync wrapper for callers that aren't async (e.g. the Streamlit app
        and AMRAGOrchestrator.run, which stay synchronous end-to-end)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (typical for Streamlit/CLI) -- safe to use asyncio.run
            return asyncio.run(self.retrieve(lesion_findings, top_k))

        # A loop is already running (e.g. inside Jupyter) -- run in a
        # separate thread with its own loop instead of nesting asyncio.run.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.retrieve(lesion_findings, top_k))
            return future.result()

    @staticmethod
    def _dedupe(chunks: list[EvidenceChunk], threshold: float = 0.9) -> list[EvidenceChunk]:
        """Drop near-duplicate sentences (e.g. the same guideline text appearing
        in both a local document and a PubMed abstract)."""
        kept: list[EvidenceChunk] = []
        seen_texts: list[set[str]] = []
        for chunk in chunks:
            words = set(chunk.text.lower().split())
            is_dup = any(
                len(words & prev) / max(len(words | prev), 1) > threshold
                for prev in seen_texts
            )
            if not is_dup:
                kept.append(chunk)
                seen_texts.append(words)
        return kept
