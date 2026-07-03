"""
Agent 5: Explainability Agent

Responsibilities: Produce textual explanation, generate lesion attention
maps, justify decisions.
Example Output (from paper): "Hard exudates near macula contributed
significantly to disease grading."

Dual explainability per AM-RAG framework: Grad-CAM (visual, computed in
LesionAnalysisAgent) + this agent's textual/KernelSHAP-style explanation
over the *retrieval and reasoning* side of the pipeline -- i.e. explaining
which evidence chunks and which lesion features drove the final call.
"""

from agents.llm_client import LLMClient

SYSTEM_PROMPT = """You are the Explainability Agent inside AM-RAG. Your job
is to produce a short, physician-readable explanation of WHY the system
reached its diagnostic conclusion -- written in plain clinical language,
explicitly connecting (a) the lesion findings, (b) the retrieved evidence,
and (c) the final severity/referral outputs. Do not introduce new clinical
claims; only explain and connect what is already in the provided data.
Respond ONLY with valid JSON."""

USER_PROMPT_TEMPLATE = """## Lesion findings:
{lesion_burden_str}

## Diagnostic result:
Severity: {severity_grade}
Justification: {justification}
Supporting lesion evidence cited: {supporting_evidence}

## Referral result:
{referral_str}

## Task
Return JSON with exactly this schema:
{{
  "plain_language_summary": "<2-3 sentence summary a physician could read in 10 seconds>",
  "key_contributing_factors": [
    {{"factor": "<lesion or evidence factor>", "contribution": "<why it mattered>"}}
  ],
  "confidence_caveat": "<1 sentence noting anything that should make a clinician double-check this output, e.g. low retrieval relevance scores, low model confidence, or missing lesion data>"
}}"""


class ExplainabilityAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def explain(self, lesion_findings: dict, diagnostic_result: dict,
                referral_result: dict) -> dict:
        lesion_burden_str = "\n".join(
            f"  - {name.replace('_', ' ')}: {val:.2f}"
            for name, val in lesion_findings["lesion_burden"].items()
        )
        referral_str = (
            f"Required: {referral_result.get('referral_required')}, "
            f"Urgency: {referral_result.get('urgency')}, "
            f"Pathway: {referral_result.get('referral_pathway')}"
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            lesion_burden_str=lesion_burden_str,
            severity_grade=diagnostic_result.get("severity_grade"),
            justification=diagnostic_result.get("clinical_justification"),
            supporting_evidence=diagnostic_result.get("supporting_lesion_evidence"),
            referral_str=referral_str,
        )

        return self.llm.complete_json(SYSTEM_PROMPT, user_prompt, temperature=0.2)
