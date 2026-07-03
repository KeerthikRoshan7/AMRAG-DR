import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import hf_hub_download

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import AMRAGOrchestrator
from utils.gradcam import overlay_gradcam
from models.lesion_detector import SEVERITY_LABELS

load_dotenv()

st.set_page_config(page_title="AM-RAG: Explainable DR Screening", layout="wide")

CHECKPOINT_PATH = os.environ.get("LESION_CHECKPOINT_PATH", "checkpoints/lesion_detector.pt")
DEVICE = os.environ.get("DEVICE", "cpu")

@st.cache_resource(show_spinner="Loading AM-RAG pipeline...")
def get_orchestrator():
    local_checkpoint = "checkpoints/best_model.pt"
    if not os.path.exists(local_checkpoint):
        try:
            model_path = hf_hub_download(
                repo_id="ROZN/AMRAG-V1", 
                filename="best_model.pt"
            )
        except Exception:
            model_path = None
    else:
        model_path = local_checkpoint
        
    return AMRAGOrchestrator(checkpoint_path=model_path, device=DEVICE)

# --- App Layout ---
st.title("🩺 AM-RAG: Agentic Multimodal RAG for Diabetic Retinopathy Screening")

# Sidebar for metadata
with st.sidebar:
    st.header("Patient Metadata")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    dtype = st.selectbox("Diabetes type", ["Not specified", "Type 1", "Type 2", "Gestational"])
    duration = st.number_input("Diabetes duration (years)", min_value=0, max_value=80, value=0)
    hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

uploaded_file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

# Use session state to persist results across reruns
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create columns for initial display
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Uploaded fundus image", width='stretch')

    if st.button("Run AM-RAG Analysis", type="primary"):
        metadata = {k: v for k, v in {
            "age": age if age > 0 else None,
            "diabetes_type": dtype if dtype != "Not specified" else None,
            "diabetes_duration_years": duration if duration > 0 else None,
            "hba1c": hba1c if hba1c > 0 else None
        }.items() if v is not None}

        orchestrator = get_orchestrator()
        with st.spinner("Agentic reasoning in progress..."):
            st.session_state.analysis_results = orchestrator.run(image, patient_metadata=metadata or None)
            
            # --- LOGGING FOR STREAMLIT CLOUD ---
            report, _, _ = st.session_state.analysis_results
            print("\n" + "="*40)
            print(f" ANALYSIS LOG - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*40)
            print(f"Model Predicted: {report['lesion_findings']['severity_label']}")
            print(f"Model Confidence: {report['lesion_findings']['severity_confidence']:.4f}")
            print(f"Agent Reviewed:  {report['diagnosis']['severity_grade']}")
            print(f"Agent Confidence: {report['diagnosis']['confidence_score']:.4f}")
            print(f"Referral Req:    {report['referral']['referral_required']}")
            print(f"Total Time:      {sum(report['timings'].values()):.2f}s")
            print("="*40 + "\n")

    # Rendering section: Display results if they exist in session state
    if st.session_state.analysis_results:
        report, gradcam_map, lesion_gradcam_maps = st.session_state.analysis_results

        with c2:
            st.subheader("Grad-CAM Analysis")
            view_options = ["Overall severity"] + (
                [name.replace("_", " ").title() for name in (lesion_gradcam_maps or {})]
            )
            selected_view = st.selectbox("Localize by", view_options)

            if selected_view == "Overall severity":
                cam_to_show = gradcam_map
            else:
                key = selected_view.lower().replace(" ", "_")
                cam_to_show = (lesion_gradcam_maps or {}).get(key)

            if cam_to_show is not None:
                overlay = overlay_gradcam(image, cam_to_show)
                st.image(overlay, caption=f"Grad-CAM++ attention overlay ({selected_view})", width='stretch')
            else:
                st.warning("Grad-CAM map could not be generated.")

        # Metrics and Results
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model-predicted severity", report["lesion_findings"]["severity_label"])
        col2.metric("Agent-reviewed severity", report["diagnosis"]["severity_grade"])
        col3.metric("Diagnostic confidence", f"{report['diagnosis']['confidence_score']:.0%}")

        st.subheader("📋 Lesion Analysis")
        st.bar_chart(report["lesion_findings"]["lesion_burden"])

        st.subheader("🧠 Diagnostic Reasoning")
        st.write(report["diagnosis"]["clinical_justification"])
        st.caption(
            f"Evidence verification: "
            f"{report['diagnosis']['evidence_verification']['claims_supported_by_evidence']}/"
            f"{report['diagnosis']['evidence_verification']['claims_checked']} claims grounded."
        )

        st.subheader("📚 Retrieved Clinical Evidence")
        for i, ev in enumerate(report["retrieved_evidence"]):
            badge = ev.get("source_type", "local").upper()
            meta_bits = [b for b in [ev.get("journal"), str(ev.get("year") or "")] if b]
            meta_str = f" · {' · '.join(meta_bits)}" if meta_bits else ""
            header = f"[{i+1}] {badge}{meta_str} (relevance: {ev.get('relevance_score', 0):.2f})"
            with st.expander(header):
                st.write(f"“{ev['text']}”")
                if ev.get("source_type", "local") != "local":
                    st.markdown(f"[{ev.get('title', 'View source')}]({ev['source_url']})")
                else:
                    st.caption(f"Source: {ev.get('title', ev.get('source_url'))}")

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
