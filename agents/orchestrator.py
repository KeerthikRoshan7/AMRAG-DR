"""
Agentic Clinical Reasoning Engine orchestrator (AM-RAG Section 5.1, module 5).

Runs the full pipeline (Algorithm 1, Section 5.9):
  1. Preprocess image            -> handled by LesionAnalysisAgent internally
  2. Detect lesions              -> LesionAnalysisAgent
  3. Extract visual features     -> LesionAnalysisAgent (fused_features)
  4. Generate clinical query     -> EvidenceRetrievalAgent
  5. Retrieve Top-K documents    -> EvidenceRetrievalAgent
  6. Fuse image + text features  -> (implicit: passed together into reasoning)
  7. Agentic clinical reasoning  -> DiagnosticReasoningAgent + ReferralAgent
  8. Generate explainable report -> ExplainabilityAgent + report assembly
  9. Provide referral recommendation -> ReferralAgent (step 7)

Returns a single structured report dict consumed by the API / Streamlit app.
"""

import time
from PIL import Image

from agents.lesion_analysis_agent import LesionAnalysisAgent
from agents.evidence_retrieval_agent import EvidenceRetrievalAgent
from agents.diagnostic_reasoning_agent import DiagnosticReasoningAgent
from agents.referral_agent import ReferralAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.llm_client import LLMClient


class AMRAGOrchestrator:
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        llm_client = LLMClient()  # shared Groq client across all LLM agents

        self.lesion_agent = LesionAnalysisAgent(checkpoint_path=checkpoint_path, device=device)
        self.retrieval_agent = EvidenceRetrievalAgent()
        self.diagnostic_agent = DiagnosticReasoningAgent(llm_client=llm_client)
        self.referral_agent = ReferralAgent(llm_client=llm_client)
        self.explainability_agent = ExplainabilityAgent(llm_client=llm_client)

    def run(self, image: Image.Image, patient_metadata: dict | None = None,
            top_k_evidence: int = 5) -> dict:
        timings = {}

        t0 = time.time()
        lesion_findings = self.lesion_agent.analyze(image)
        timings["lesion_analysis_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        evidence = self.retrieval_agent.retrieve(lesion_findings, top_k=top_k_evidence)
        timings["evidence_retrieval_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        diagnostic_result = self.diagnostic_agent.reason(
            lesion_findings, evidence, patient_metadata=patient_metadata
        )
        timings["diagnostic_reasoning_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        referral_result = self.referral_agent.recommend(diagnostic_result)
        timings["referral_recommendation_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        explanation = self.explainability_agent.explain(
            lesion_findings, diagnostic_result, referral_result
        )
        timings["explainability_s"] = round(time.time() - t0, 2)

        # Grad-CAM map is a numpy array -- keep separate from the JSON-safe report
        gradcam_map = lesion_findings.pop("gradcam_map", None)
        lesion_findings.pop("fused_features", None)  # not JSON-serializable, internal only

        report = {
            "model_checkpoint_status": "trained" if lesion_findings.pop("is_trained_checkpoint") else "UNTRAINED_DEMO_MODE",
            "lesion_findings": lesion_findings,
            "retrieved_evidence": evidence,
            "diagnosis": {
                k: v for k, v in diagnostic_result.items() if not k.startswith("_")
            },
            "referral": referral_result,
            "explanation": explanation,
            "timings": timings,
        }
        return report, gradcam_map
