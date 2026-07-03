"""
Agent 4: Referral Recommendation Agent

Responsibilities: Determine urgency, generate referral pathway, suggest
follow-up schedule.
Example Output (from paper): "Referral Required: Within 3 Months."
"""

from agents.llm_client import LLMClient
from agents.evidence_formatting import format_evidence

SYSTEM_PROMPT = """You are the Referral Recommendation Agent inside AM-RAG.
Given a diagnostic reasoning result and its supporting evidence, determine:
1. Whether referral to an eye care professional / retina specialist is
   warranted.
2. The urgency/timeframe, grounded in the retrieved follow-up/referral
   evidence chunks (do not invent a timeframe that isn't supported).
3. A brief referral pathway description.

Respond ONLY with valid JSON."""

USER_PROMPT_TEMPLATE = """## Diagnostic Reasoning Agent output:
Severity: {severity_grade}
Confidence: {confidence_score}
Justification: {justification}

## Retrieved evidence available for grounding referral timing:
{evidence_str}

## Task
Return JSON with exactly this schema:
{{
  "referral_required": <true/false>,
  "urgency": "<one of: routine (12 months), routine (6-9 months), prompt (1-3 months), urgent (same week), emergent (same day)>",
  "referral_pathway": "<who to refer to, e.g. 'Optometrist for continued screening' or 'Retina specialist for PRP/anti-VEGF evaluation'>",
  "rationale": "<1-2 sentences citing the specific evidence supporting this timeframe>"
}}"""


class ReferralAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def recommend(self, diagnostic_result: dict) -> dict:
        evidence = diagnostic_result.get("_retrieved_evidence", [])
        evidence_str = format_evidence(evidence)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            severity_grade=diagnostic_result.get("severity_grade"),
            confidence_score=diagnostic_result.get("confidence_score"),
            justification=diagnostic_result.get("clinical_justification"),
            evidence_str=evidence_str,
        )

        return self.llm.complete_json(SYSTEM_PROMPT, user_prompt, temperature=0.1)
