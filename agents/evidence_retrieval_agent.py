"""
Agent 2: Evidence Retrieval Agent (AM-RAG Section 5.3.1)

Responsibilities:
1. Generate a clinical query based on visual findings.
2. Retrieve relevant evidence from the ophthalmology knowledge base.
"""

import asyncio
from knowledge_base.web_retriever import WebEvidenceRetriever


class EvidenceRetrievalAgent:
    def __init__(self):
        # We don't load the embedder here to keep it lightweight, 
        # WebEvidenceRetriever will use lexical fallback or we can pass one if needed.
        self.retriever = WebEvidenceRetriever()

    async def retrieve(self, lesion_findings: dict, top_k: int = 5) -> list[dict]:
        """
        Generate a query from lesion findings and retrieve evidence.
        """
        # Create a clinical query from findings
        severity = lesion_findings.get("severity_label", "diabetic retinopathy")
        lesions = [name.replace("_", " ") for name, val in lesion_findings.get("lesion_burden", {}).items() if val > 0.2]
        
        query = f"diabetic retinopathy {severity}"
        if lesions:
            query += " with " + ", ".join(lesions)
            
        # Perform retrieval
        evidence_chunks = await self.retriever.retrieve(query, top_k=top_k)
        return [chunk.to_dict() for chunk in evidence_chunks]

    def retrieve_sync(self, lesion_findings: dict, top_k: int = 5) -> list[dict]:
        """Synchronous wrapper for retrieve()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If we're already in an event loop (e.g. FastAPI/Streamlit), 
            # we might need a different approach, but for now this is standard.
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.retrieve(lesion_findings, top_k=top_k))
        else:
            return loop.run_until_complete(self.retrieve(lesion_findings, top_k=top_k))
