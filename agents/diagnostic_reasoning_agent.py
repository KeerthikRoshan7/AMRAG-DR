"""
Agent 3: Diagnostic Reasoning Agent

The central intelligence component. Implements the five-stage multimodal
reasoning workflow:
  Stage 1: Evidence Aggregation      (C = P u R)
  Stage 2: Knowledge Alignment       (compare findings against ICDR/AAO criteria)
  Stage 3: Contextual Reasoning      (LLM reasoning over unified clinical context)
  Stage 4: Evidence Verification     (verification score per clinical claim)
  Stage 5: Diagnosis Generation      (severity grade, confidence, justification)

D = f(P, R, M) where P = pathology vector, R = retrieved evidence,
M = patient metadata.
"""

from agents.llm_client import LLMClient
from agents.evidence_formatting import format_evidence

SYSTEM_PROMPT = """You are the Diagnostic Reasoning Agent inside AM-RAG, a
clinical decision-support system for diabetic retinopathy. You reason over
three inputs only: (1) quantitative lesion findings from an image analysis
model, (2) retrieved clinical evidence chunks from an ophthalmology
knowledge base, and (3) optional patient metadata.

Rules:
- Every clinical claim you make MUST be traceable to either the lesion
  findings or a specific retrieved evidence chunk. Do not introduce facts
  that appear in neither.
- If the retrieved evidence does not clearly support a claim, say so
  explicitly rather than filling the gap from general knowledge.
- This is a decision-SUPPORT tool for a physician, not an autonomous
  diagnosis. Frame conclusions as findings and recommendations for
  physician review, not as final diagnoses.
- Respond ONLY with valid JSON matching the requested schema."""

USER_PROMPT_TEMPLATE = """## Stage 1-2: Clinical Context (P union R)

### Lesion findings (P) from Lesion Analysis Agent:
Severity grade predicted by model: {severity_label} (confidence: {confidence:.1%})
Lesion burden (normalized 0-1 scale):
{lesion_burden_str}

### Retrieved clinical evidence (R) from knowledge base:
{evidence_str}

### Patient metadata (M):
{metadata_str}

## Task
Perform Stage 3 (contextual reasoning), Stage 4 (evidence verification), and
Stage 5 (diagnosis generation). Return JSON with exactly this schema:

{{
  "severity_grade": "<one of: No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR>",
  "confidence_score": <0-1 float, your own confidence after evidence review, may differ from model confidence>,
  "supporting_lesion_evidence": ["<short phrase citing specific lesion findings>", ...],
  "clinical_justification": "<2-4 sentences explaining the severity call, explicitly referencing which retrieved evidence chunk(s) support the ICDR criteria applied>",
  "evidence_verification": {{
    "claims_checked": <int>,
    "claims_supported_by_evidence": <int>,
    "verification_notes": "<brief note on any claim you could NOT fully ground in retrieved evidence>"
  }},
  "follow_up_recommendation": "<short recommendation, grounded in retrieved evidence about follow-up intervals>"
}}"""


class DiagnosticReasoningAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def reason(self, lesion_findings: dict, evidence: list[dict],
               patient_metadata: dict | None = None) -> dict:
        lesion_burden_str = "\n".join(
            f"  - {name.replace('_', ' ')}: {val:.2f}"
            for name, val in lesion_findings["lesion_burden"].items()
        )
        evidence_str = format_evidence(evidence)

        metadata_str = "\n".join(f"  - {k}: {v}" for k, v in (patient_metadata or {}).items()) \
            or "  Not provided."

        user_prompt = USER_PROMPT_TEMPLATE.format(
            severity_label=lesion_findings["severity_label"],
            confidence=lesion_findings["severity_confidence"],
            lesion_burden_str=lesion_burden_str,
            evidence_str=evidence_str,
            metadata_str=metadata_str,
        )

        result = self.llm.complete_json(SYSTEM_PROMPT, user_prompt, temperature=0.1)
        result["_model_predicted_severity"] = lesion_findings["severity_label"]
        result["_retrieved_evidence"] = evidence
        return result
