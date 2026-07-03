"""Single source of truth for turning an evidence dict (as produced by
EvidenceChunk.to_dict()) into the citation string used in LLM prompts.
Do not inline this formatting logic elsewhere -- import format_evidence.
"""

def format_evidence(evidence: list[dict]) -> str:
    def _fmt(i: int, e: dict) -> str:
        source = e.get("source") or e.get("title") or e.get("source_url", "unknown source")
        score = e.get("score", e.get("relevance_score", 0.0))
        text = e.get("text", "")
        return f"[{i+1}] (source: {source}, relevance: {score:.2f})\n{text}"

    return "\n\n".join(_fmt(i, e) for i, e in enumerate(evidence)) or "No evidence retrieved."
