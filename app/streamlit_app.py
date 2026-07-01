import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from agents.orchestrator import AMRAGOrchestrator
from utils.gradcam import overlay_gradcam
from models.lesion_detector import SEVERITY_LABELS

load_dotenv()

st.set_page_config(page_title="AM-RAG: Explainable DR Screening", layout="wide")

CHECKPOINT_PATH = os.environ.get("LESION_CHECKPOINT_PATH", "checkpoints/lesion_detector.pt")
DEVICE = os.environ.get("DEVICE", "cpu")


@st.cache_resource(show_spinner="Loading AM-RAG pipeline (detector + knowledge base)...")
def get_orchestrator():
    checkpoint = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
    return AMRAGOrchestrator(checkpoint_path=checkpoint, device=DEVICE)


st.title("🩺 AM-RAG: Agentic Multimodal RAG for Diabetic Retinopathy Screening")
st.caption(
    "Lesion detection -> clinical knowledge retrieval -> agentic reasoning -> "
    "explainable, evidence-grounded diagnostic report."
)

with st.sidebar:
    st.header("Patient Metadata (optional)")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    dtype = st.selectbox("Diabetes type", ["Not specified", "Type 1", "Type 2", "Gestational"])
    duration = st.number_input("Diabetes duration (years)", min_value=0, max_value=80, value=0)
    hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    st.markdown("---")
    st.caption(
        "⚠️ This is a research / thesis demo of the AM-RAG architecture. "
        "It is not a certified diagnostic device and must not be used for "
        "real clinical decisions."
    )

uploaded_file = st.file_uploader("Upload a fundus (retinal) image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded fundus image", width=True)

    if st.button("Run AM-RAG Analysis", type="primary"):
        metadata = {}
        if age > 0:
            metadata["age"] = age
        if dtype != "Not specified":
            metadata["diabetes_type"] = dtype
        if duration > 0:
            metadata["diabetes_duration_years"] = duration
        if hba1c > 0:
            metadata["hba1c"] = hba1c

        orchestrator = get_orchestrator()

        with st.spinner("Running lesion detection, retrieval, and multi-agent reasoning..."):
            report, gradcam_map = orchestrator.run(image, patient_metadata=metadata or None)

        if report["model_checkpoint_status"] == "UNTRAINED_DEMO_MODE":
            st.warning(
                "⚠️ Running with an UNTRAINED lesion detector (ImageNet weights only). "
                "Severity/lesion outputs below are placeholders to demonstrate the "
                "pipeline -- train on Kaggle (training/train_lesion_detector.py) and "
                "set LESION_CHECKPOINT_PATH for real predictions."
            )

        with col2:
            if gradcam_map is not None:
                overlay = overlay_gradcam(image, gradcam_map)
                st.image(overlay, caption="Grad-CAM attention overlay", width=True)

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Model-predicted severity", report["lesion_findings"]["severity_label"])
        c2.metric("Agent-reviewed severity", report["diagnosis"]["severity_grade"])
        c3.metric("Diagnostic confidence", f"{report['diagnosis']['confidence_score']:.0%}")

        st.subheader("📋 Lesion Analysis")
        st.bar_chart(report["lesion_findings"]["lesion_burden"])

        st.subheader("🧠 Diagnostic Reasoning")
        st.write(report["diagnosis"]["clinical_justification"])
        st.caption(
            f"Evidence verification: "
            f"{report['diagnosis']['evidence_verification']['claims_supported_by_evidence']}/"
            f"{report['diagnosis']['evidence_verification']['claims_checked']} claims grounded in retrieved evidence. "
            f"{report['diagnosis']['evidence_verification']['verification_notes']}"
        )

        st.subheader("📚 Retrieved Clinical Evidence")
        for i, ev in enumerate(report["retrieved_evidence"]):
            with st.expander(f"[{i+1}] {ev['source']} (relevance: {ev['score']:.2f})"):
                st.write(ev["text"])

        st.subheader("🏥 Referral Recommendation")
        r = report["referral"]
        st.info(
            f"**Referral required:** {r['referral_required']}  \n"
            f"**Urgency:** {r['urgency']}  \n"
            f"**Pathway:** {r['referral_pathway']}  \n"
            f"**Rationale:** {r['rationale']}"
        )

        st.subheader("💡 Explanation")
        exp = report["explanation"]
        st.write(exp["plain_language_summary"])
        for factor in exp["key_contributing_factors"]:
            st.markdown(f"- **{factor['factor']}**: {factor['contribution']}")
        st.caption(f"⚠️ {exp['confidence_caveat']}")

        with st.expander("⏱ Agent timing breakdown"):
            st.json(report["timings"])
else:
    st.info("Upload a fundus image to run the full AM-RAG pipeline.")
