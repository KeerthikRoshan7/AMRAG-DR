import os
import sys
import time
import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, roc_auc_score

# Try to import the image comparison component
try:
    from streamlit_image_comparison import image_comparison
    HAS_IMAGE_COMPARISON = True
except ImportError:
    HAS_IMAGE_COMPARISON = False

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock imports for environment where actual modules might not be present
# In a real scenario, these would be the actual imports
try:
    from agents.orchestrator import AMRAGOrchestrator
    from utils.gradcam import overlay_gradcam
    from models.lesion_detector import SEVERITY_LABELS
except ImportError:
    # Fallback for demonstration/development if the full repo isn't cloned
    class AMRAGOrchestrator:
        def __init__(self, **kwargs): pass
        def run(self, image, **kwargs): return {}, None, {}
    def overlay_gradcam(img, cam): return img
    SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

load_dotenv()

st.set_page_config(
    page_title="AM-RAG: Explainable DR Screening", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX improvements
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .referral-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e0f2fe;
        border-left: 5px solid #0369a1;
    }
    </style>
    """, unsafe_allow_html=True)

CHECKPOINT_PATH = os.environ.get("LESION_CHECKPOINT_PATH", "checkpoints/lesion_detector.pt")
DEVICE = os.environ.get("DEVICE", "cpu")

# --- APTOS 2019 Defaults ---
DEFAULT_KAGLLE_DATASET = "mariaherrerot/aptos2019"
DEFAULT_CSV = "train_1.csv"
DEFAULT_IMG_DIR = "train_images/train_images"

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
st.title("🩺 AM-RAG: Explainable Diabetic Retinopathy Screening")
st.markdown("### Agentic Multimodal RAG for Clinical Decision Support")

# Sidebar for metadata and Kaggle-based evaluation
with st.sidebar:
    st.header("👤 Patient Metadata")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    dtype = st.selectbox("Diabetes type", ["Not specified", "Type 1", "Type 2", "Gestational"])
    duration = st.number_input("Diabetes duration (years)", min_value=0, max_value=80, value=0)
    hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    st.markdown("---")
    st.header("⚙️ Admin: Evaluation")
    st.write("Run a background evaluation using the APTOS 2019 dataset.")
    
    if st.button("🚀 Run APTOS 2019 Benchmark"):
        # Pull from Streamlit Secrets
        k_user = st.secrets.get("KAGGLE_USERNAME")
        k_key = st.secrets.get("KAGGLE_KEY")
        
        if k_user and k_key:
            with st.spinner("Pipelining APTOS 2019 data from Kaggle..."):
                try:
                    os.environ['KAGGLE_USERNAME'] = k_user
                    os.environ['KAGGLE_KEY'] = k_key
                    
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                    
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Download APTOS 2019
                        api.dataset_download_files(DEFAULT_KAGLLE_DATASET, path=tmpdir, unzip=True)
                        
                        csv_path = os.path.join(tmpdir, DEFAULT_CSV)
                        img_root = os.path.join(tmpdir, DEFAULT_IMG_DIR)
                        
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path).head(24) # Reduced sample size
                            orchestrator = get_orchestrator()
                            results = []
                            
                            for _, row in df.iterrows():
                                time.sleep(1)
                                img_id = str(row["id_code"])
                                img_path = os.path.join(img_root, img_id + ".png")
                                if not os.path.exists(img_path):
                                    img_path = os.path.join(img_root, img_id + ".jpg")
                                
                                if os.path.exists(img_path):
                                    image = Image.open(img_path).convert("RGB")
                                    report, _, _ = orchestrator.run(image)
                                    results.append({
                                        "target": int(row["diagnosis"]),
                                        "vision_pred": report["lesion_findings"]["severity_grade"],
                                        "vision_probs": [report["lesion_findings"]["severity_probs"][l] for l in SEVERITY_LABELS],
                                        "reasoning_pred": SEVERITY_LABELS.index(report["diagnosis"]["severity_grade"]) if report["diagnosis"]["severity_grade"] in SEVERITY_LABELS else -1
                                    })
                            
                            if results:
                                res_df = pd.DataFrame(results)
                                targets = res_df["target"].values
                                vision_preds = res_df["vision_pred"].values
                                vision_probs = np.array(res_df["vision_probs"].tolist())
                                reasoning_preds = res_df["reasoning_pred"].values

                                vis_acc = accuracy_score(targets, vision_preds)
                                try:
                                    vis_auroc = roc_auc_score(targets, vision_probs, multi_class='ovr', average='macro')
                                except:
                                    vis_auroc = 0.0
                                
                                valid_reasoning = reasoning_preds != -1
                                reas_acc = accuracy_score(targets[valid_reasoning], reasoning_preds[valid_reasoning]) if valid_reasoning.any() else 0.0

                                st.success(f"Benchmark complete! Vision Acc: {vis_acc:.2f}, Reasoning Acc: {reas_acc:.2f}")
                            else:
                                st.error("No valid images found.")
                        else:
                            st.error(f"CSV file '{DEFAULT_CSV}' not found.")
                except Exception as e:
                    st.error(f"Kaggle benchmark failed: {e}")
        else:
            st.error("Kaggle credentials not found in Streamlit Secrets.")

# --- Main App ---
uploaded_file = st.file_uploader("📤 Upload a fundus image", type=["png", "jpg", "jpeg"])

# Use session state to persist results across reruns
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Analysis button
    if st.button("🔍 Run AM-RAG Analysis", type="primary"):
        metadata = {k: v for k, v in {
            "age": age if age > 0 else None,
            "diabetes_type": dtype if dtype != "Not specified" else None,
            "diabetes_duration_years": duration if duration > 0 else None,
            "hba1c": hba1c if hba1c > 0 else None
        }.items() if v is not None}

        orchestrator = get_orchestrator()
        with st.spinner("🧠 Agentic reasoning in progress..."):
            st.session_state.analysis_results = orchestrator.run(image, patient_metadata=metadata or None)

    # Rendering section: Display results if they exist in session state
    if st.session_state.analysis_results:
        report, gradcam_map, lesion_gradcam_maps = st.session_state.analysis_results

        # --- Image Comparison Section ---
        st.markdown("---")
        st.subheader("🖼️ Image Analysis & Localization")
        
        # Select Grad-CAM view
        view_options = ["Overall severity"] + (
            [name.replace("_", " ").title() for name in (lesion_gradcam_maps or {})]
        )
        selected_view = st.selectbox("Select localization layer:", view_options)

        if selected_view == "Overall severity":
            cam_to_show = gradcam_map
        else:
            key = selected_view.lower().replace(" ", "_")
            cam_to_show = (lesion_gradcam_maps or {}).get(key)

        if cam_to_show is not None:
            overlay = overlay_gradcam(image, cam_to_show)
            
            # Layout Choice: Side-by-side or Slider
            display_mode = st.radio("Display mode:", ["Before-After Slider", "Side-by-Side"], horizontal=True)
            
            if display_mode == "Before-After Slider" and HAS_IMAGE_COMPARISON:
                image_comparison(
                    img1=image,
                    img2=overlay,
                    label1="Original Image",
                    label2=f"Grad-CAM Overlay ({selected_view})",
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )
            else:
                if display_mode == "Before-After Slider" and not HAS_IMAGE_COMPARISON:
                    st.info("💡 `streamlit-image-comparison` not installed. Falling back to Side-by-Side view.")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="Original Fundus Image", use_container_width=True)
                with col_img2:
                    st.image(overlay, caption=f"Grad-CAM Overlay ({selected_view})", use_container_width=True)
        else:
            st.image(image, caption="Original Fundus Image", use_container_width=True)
            st.warning("Grad-CAM map could not be generated for this view.")

        # --- Metrics and Results ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Prediction", report["lesion_findings"]["severity_label"])
        with col2:
            st.metric("Agent Review", report["diagnosis"]["severity_grade"])
        with col3:
            st.metric("Confidence Score", f"{report['diagnosis']['confidence_score']:.0%}")

        # Detailed Analysis
        tab1, tab2, tab3 = st.tabs(["📊 Lesion Analysis", "🧠 Diagnostic Reasoning", "📚 Clinical Evidence"])

        with tab1:
            st.subheader("Lesion Burden Analysis")
            st.bar_chart(report["lesion_findings"]["lesion_burden"])
            
            # Summary of findings
            findings = report["lesion_findings"].get("findings_summary", "No specific summary available.")
            st.info(f"**Findings Summary:** {findings}")

        with tab2:
            st.subheader("Clinical Justification")
            st.write(report["diagnosis"]["clinical_justification"])
            st.caption(
                f"✅ Evidence verification: "
                f"{report['diagnosis']['evidence_verification']['claims_supported_by_evidence']}/"
                f"{report['diagnosis']['evidence_verification']['claims_checked']} claims grounded."
            )
            
            # Referral Recommendation in a stylized box
            st.markdown("### 🏥 Referral Recommendation")
            r = report["referral"]
            st.markdown(f"""
            <div class="referral-box">
                <strong>Referral Required:</strong> {r['referral_required']}<br>
                <strong>Urgency:</strong> {r['urgency']}<br>
                <strong>Pathway:</strong> {r['referral_pathway']}<br>
                <strong>Rationale:</strong> {r['rationale']}
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.subheader("Retrieved Clinical Evidence")
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

        # Explanation Section
        with st.expander("💡 Patient-Friendly Explanation"):
            exp = report["explanation"]
            st.write(exp["plain_language_summary"])
            st.markdown("#### Key Contributing Factors")
            for factor in exp["key_contributing_factors"]:
                st.markdown(f"- **{factor['factor']}**: {factor['contribution']}")
            
            st.markdown("#### Suggested Next Steps")
            for step in exp["next_steps"]:
                st.markdown(f"- {step}")

    else:
        # Initial state before analysis
        st.info("Please upload a fundus image and click 'Run AM-RAG Analysis' to begin.")
        st.image(image, caption="Uploaded fundus image", use_container_width=True)

else:
    # Welcome screen / Instructions
    st.info("👋 Welcome! Please upload a retinal fundus image in the sidebar or main area to start the AI-assisted screening process.")
    
    # Placeholder for UI balance
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image("https://raw.githubusercontent.com/KeerthikRoshan7/DR-MARVEL/main/assets/logo.png", use_container_width=True) # Assuming a logo exists or just skip
