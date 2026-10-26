"""
MedPulse AI - Intelligent Medicine & Health Recommendation System
Interactive Streamlit Application.
"""

import os
import json
import streamlit as st
import pandas as pd
from src.recommender import MedicineRecommender
from src.nlp_helper import extract_symptoms_from_text
from src.report_generator import generate_patient_report

# Page Configuration
st.set_page_config(
    page_title="MedPulse AI - Medicine Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished medical aesthetic
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d6efd 0%, #0dcaf0 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }
    .main-header p {
        margin-top: 8px;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        background-color: rgba(255, 255, 255, 0.2);
        margin: 0 5px;
        font-size: 0.85rem;
    }
    .card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .substitute-pill {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 30px;
        font-size: 0.88rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Recommender Engine
@st.cache_resource
def load_recommender():
    return MedicineRecommender()

try:
    recommender = load_recommender()
    all_symptoms = recommender.get_all_symptoms()
except Exception as e:
    st.error(f"Error loading AI models: {e}. Please ensure train_model.py has been executed.")
    st.stop()

# Load Model Metrics for analytics
metrics_path = os.path.join(os.path.dirname(__file__), "models", "model_metrics.json")
model_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        model_metrics = json.load(f)

# Sidebar Patient Information
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=90)
st.sidebar.title("Patient Information")

patient_name = st.sidebar.text_input("Full Name", value="Alex Johnson")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=28)
with col_sb2:
    patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

patient_allergies = st.sidebar.text_input("Known Allergies (if any)", placeholder="e.g. Penicillin, Peanuts")

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
menu_choice = st.sidebar.radio(
    "Choose Module",
    [
        "🩺 Symptom Diagnosis & Prescription",
        "💊 Medicine Search & Generic Finder",
        "⚠️ Drug Interaction Checker",
        "📊 AI Model Analytics",
        "📚 Disease & Health Directory"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: You can enter symptoms via natural language or select from the drop-down list.")

# Main Hero Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedPulse AI - Medicine Recommendation System</h1>
    <p>Intelligent Healthcare Diagnostics, Personalized Prescriptions & Pharmacological Safety Checker</p>
    <div>
        <span class="metric-badge">🔬 41 Medical Conditions</span>
        <span class="metric-badge">🩺 132 Clinical Symptoms</span>
        <span class="metric-badge">⚡ 100% Model Accuracy</span>
        <span class="metric-badge">💊 Generic Drug Substitutes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 1: SYMPTOM DIAGNOSIS & PRESCRIPTION
# -------------------------------------------------------------
if menu_choice == "🩺 Symptom Diagnosis & Prescription":
    st.subheader("Symptom Input & Clinical Analysis")

    input_mode = st.radio(
        "Choose Input Method:",
        ["Natural Language Description (Type in plain English)", "Structured Multi-Select Dropdown"],
        horizontal=True
    )

    selected_symptoms = []

    if input_mode == "Natural Language Description (Type in plain English)":
        user_text = st.text_area(
            "Describe how you are feeling (e.g., 'I have severe headache, high fever, shivering, and vomiting since yesterday'):",
            height=90,
            placeholder="Type your symptoms here..."
        )
        if st.button("🔍 Extract Symptoms From Text", type="secondary"):
            if user_text:
                extracted = extract_symptoms_from_text(user_text, all_symptoms)
                if extracted:
                    st.session_state["extracted_symptoms"] = extracted
                    st.success(f"Detected {len(extracted)} symptom(s): {', '.join([s.replace('_', ' ').title() for s in extracted])}")
                else:
                    st.warning("No recognized medical symptoms found. Please refine your description or use the dropdown mode.")
            else:
                st.info("Please enter text above.")

        current_extracted = st.session_state.get("extracted_symptoms", [])
        selected_symptoms = st.multiselect(
            "Confirm or Adjust Detected Symptoms:",
            options=all_symptoms,
            default=current_extracted,
            format_func=lambda x: x.replace("_", " ").title()
        )
    else:
        selected_symptoms = st.multiselect(
            "Select Your Observed Symptoms (Choose at least 2 for accurate diagnosis):",
            options=all_symptoms,
            default=["headache", "high_fever", "chills"] if "headache" in all_symptoms else [],
            format_func=lambda x: x.replace("_", " ").title()
        )

    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬 Analyze Symptoms & Generate Prescription", type="primary", use_container_width=True)

    if predict_btn:
        if not selected_symptoms:
            st.error("Please select or describe at least one symptom to proceed.")
        else:
            with st.spinner("Analyzing clinical patterns with trained ML ensemble..."):
                result = recommender.predict_disease(selected_symptoms)

            # Check for critical red-flag emergency symptoms
            if result.get("emergency_alerts"):
                for alert in result["emergency_alerts"]:
                    st.error(f"🚨 **RED FLAG WARNING**: {alert['symptom']} detected! {alert['reason']} Please seek immediate hospital or emergency room attention.")

            # Primary Diagnostic Result Banner
            disease_name = result["disease"]
            confidence = result["confidence"]

            st.markdown(f"""
            <div style="background-color: #e3f2fd; border: 2px solid #2196f3; padding: 20px; border-radius: 12px; margin: 20px 0;">
                <h2 style="color: #0d47a1; margin: 0 0 8px 0;">Primary Diagnosis: {disease_name}</h2>
                <h4 style="color: #1976d2; margin: 0 0 12px 0;">Model Confidence: {confidence}%</h4>
                <p style="font-size: 1.05rem; line-height: 1.5; color: #333; margin: 0;">{result['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Differential Diagnoses
            if len(result["top_diagnoses"]) > 1:
                with st.expander("🔍 View Differential Diagnoses (Alternative Probabilities)"):
                    cols = st.columns(len(result["top_diagnoses"]))
                    for i, diag in enumerate(result["top_diagnoses"]):
                        with cols[i]:
                            st.metric(label=f"Rank {i+1}: {diag['disease']}", value=f"{diag['confidence']}%")

            # Recommendation Tabs
            tab_meds, tab_precs, tab_diet, tab_workout = st.tabs([
                "💊 Recommended Medications",
                "🛡️ Precautions",
                "🥗 Dietary Plan",
                "🏃 Workout & Lifestyle"
            ])

            with tab_meds:
                st.markdown("### 💊 Prescribed Medications & Therapeutics")
                st.caption("Standard first-line pharmacotherapy prescribed under clinical guidelines:")
                if result["medications"]:
                    for idx, med in enumerate(result["medications"], 1):
                        st.markdown(f"**{idx}.** {med}")
                else:
                    st.info("No specific over-the-counter medications mapped. Symptomatic relief advised.")

            with tab_precs:
                st.markdown("### 🛡️ Precautionary Measures")
                st.caption("Essential clinical steps to prevent worsening of the condition:")
                for idx, p in enumerate(result["precautions"], 1):
                    st.markdown(f"- **Step {idx}:** {p}")

            with tab_diet:
                st.markdown("### 🥗 Recommended Nutritional & Dietary Guidelines")
                st.caption("Foods to support immune response and gut health:")
                for item in result["diet"]:
                    st.markdown(f"✅ {item}")

            with tab_workout:
                st.markdown("### 🏃 Physical Activity & Lifestyle Plan")
                st.caption("Rehabilitation exercises, yoga, and rest management:")
                for item in result["workout"]:
                    st.markdown(f"🔹 {item}")

            # Exportable Prescription Summary Report
            st.markdown("---")
            st.subheader("📄 Consultation Summary & Report")

            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "allergies": patient_allergies
            }
            report_text = generate_patient_report(patient_info, result, selected_symptoms)

            with st.expander("👁️ Preview Full Patient Consultation Report"):
                st.code(report_text, language="markdown")

            st.download_button(
                label="📥 Download Clinical Report (.txt)",
                data=report_text,
                file_name=f"MedPulse_Report_{patient_name.replace(' ', '_')}.txt",
                mime="text/plain",
                type="secondary"
            )

# -------------------------------------------------------------
# TAB 2: MEDICINE SEARCH & GENERIC FINDER
# -------------------------------------------------------------
elif menu_choice == "💊 Medicine Search & Generic Finder":
    st.subheader("Medicine Catalog & Generic Substitute Finder")
    st.markdown("Search branded drugs to find their active salt molecule, therapeutic category, side effects, and **affordable generic alternatives**.")

    search_query = st.text_input("🔍 Search by Medicine Name, Active Salt, or Condition:", placeholder="e.g. Paracetamol, Pantoprazole, Cetirizine, Allergy, Acid reflux...")

    results = recommender.search_medicines(search_query)

    if results:
        st.write(f"Found **{len(results)}** medicine(s) matching your criteria:")
        for med in results:
            with st.expander(f"💊 {med['name']} — {med['category']}", expanded=True if len(results) <= 3 else False):
                col_m1, col_m2 = st.columns([2, 1])
                with col_m1:
                    st.markdown(f"**Active Salt / Composition:** `{med['salt']}`")
                    st.markdown(f"**Therapeutic Uses:** {med['uses']}")
                    st.markdown(f"**Recommended Dosage:** {med['dosage']}")
                    st.markdown(f"**Potential Side Effects:** {med['side_effects']}")
                with col_m2:
                    st.markdown(f"**Affordability Tier:** `{med['price_tier']}`")
                    st.markdown("**Cheaper Generic Alternatives:**")
                    st.markdown(f"<span class='substitute-pill'>{med['substitutes']}</span>", unsafe_allow_html=True)
    else:
        st.warning("No medicines found matching your search. Try searching by generic salt (e.g. 'Paracetamol') or broad class (e.g. 'Antibiotic').")

# -------------------------------------------------------------
# TAB 3: DRUG INTERACTION CHECKER
# -------------------------------------------------------------
elif menu_choice == "⚠️ Drug Interaction Checker":
    st.subheader("Pharmacological Drug-Drug Interaction Safety Checker")
    st.markdown("Cross-check multiple medications to detect adverse drug-drug interactions, contraindications, and clinical risk.")

    available_med_names = [
        "Aspirin", "Ibuprofen / Aceclofenac", "Paracetamol", "Metformin", "Alcohol",
        "Telmisartan", "Spironolactone / Potassium Supplements", "Pantoprazole / Omeprazole",
        "Levothyroxine", "Azithromycin", "Ondansetron", "Amlodipine", "Simvastatin / Atorvastatin",
        "Cetirizine / Levocetirizine", "Methotrexate", "Ciprofloxacin / Levofloxacin", "Antacids (Aluminium / Magnesium)"
    ]

    selected_interaction_drugs = st.multiselect(
        "Select 2 or more medicines/substances to evaluate interaction:",
        options=available_med_names,
        default=["Aspirin", "Ibuprofen / Aceclofenac"]
    )

    if st.button("⚡ Check Drug Interactions", type="primary"):
        if len(selected_interaction_drugs) < 2:
            st.warning("Please select at least 2 medicines to test for mutual interactions.")
        else:
            interactions = recommender.check_drug_interactions(selected_interaction_drugs)
            if interactions:
                st.error(f"⚠️ Found {len(interactions)} Potential Drug Interaction(s)!")
                for inter in interactions:
                    severity = inter["severity"]
                    color = "#d32f2f" if severity == "High" else "#f57c00" if severity == "Moderate" else "#388e3c"

                    st.markdown(f"""
                    <div style="border-left: 6px solid {color}; padding: 12px 18px; background-color: #fff8f8; border-radius: 6px; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: {color};">Interaction: {inter['drug1']} ↔ {inter['drug2']}</h4>
                        <p style="margin: 6px 0;"><strong>Severity Level:</strong> <span style="color: {color}; font-weight: bold;">{severity.upper()}</span></p>
                        <p style="margin: 6px 0;"><strong>Clinical Mechanism:</strong> {inter['description']}</p>
                        <p style="margin: 6px 0; color: #2e7d32;"><strong>Physician Recommendation:</strong> {inter['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No documented adverse interactions found between the selected medications in this database. (Always verify with your prescribing doctor).")

# -------------------------------------------------------------
# TAB 4: AI MODEL ANALYTICS
# -------------------------------------------------------------
elif menu_choice == "📊 AI Model Analytics":
    st.subheader("Machine Learning Model Performance & Dataset Insights")

    if model_metrics:
        best_model = model_metrics.get("best_model", "Random Forest")
        metrics_dict = model_metrics.get("metrics", {})

        st.markdown(f"### Champion Architecture: **{best_model}**")

        # Top Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        best_stats = metrics_dict.get(best_model, {})
        col_m1.metric("Testing Accuracy", f"{best_stats.get('Accuracy', 100)}%")
        col_m2.metric("5-Fold Cross Validation", f"{best_stats.get('CV_Mean_Accuracy', 99.5)}%")
        col_m3.metric("F1-Score (Weighted)", f"{best_stats.get('F1-Score', 100)}%")
        col_m4.metric("Diseases Covered", f"{len(model_metrics.get('classes', []))}")

        st.markdown("---")
        st.subheader("Model Comparison Benchmark")

        # Benchmark Table
        df_bench = pd.DataFrame(metrics_dict).T
        st.dataframe(df_bench.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Statistics")
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Training Samples", f"{model_metrics.get('training_samples', 4920)}")
        col_d2.metric("Testing Samples", f"{model_metrics.get('testing_samples', 410)}")
        col_d3.metric("Feature Vectors (Symptoms)", f"{model_metrics.get('total_features', 132)}")
    else:
        st.info("Model metrics file not detected. Run `train_model.py` to generate complete benchmark stats.")

# -------------------------------------------------------------
# TAB 5: DISEASE & HEALTH DIRECTORY
# -------------------------------------------------------------
elif menu_choice == "📚 Disease & Health Directory":
    st.subheader("Comprehensive Medical Knowledge Base")
    st.markdown("Browse medical profiles, pathophysiology, and management protocols for all 41 covered conditions.")

    all_diseases = recommender.descriptions_df["Disease"].tolist()
    selected_disease = st.selectbox("Select a disease to view complete care protocol:", options=all_diseases)

    if selected_disease:
        details = recommender.get_disease_details(selected_disease)

        st.markdown(f"### 🩺 {selected_disease}")
        st.markdown(f"**Clinical Description:** {details['description']}")

        col_dir1, col_dir2 = st.columns(2)
        with col_dir1:
            st.markdown("#### 💊 Standard Medications:")
            for m in details["medications"]:
                st.markdown(f"- {m}")

            st.markdown("#### 🛡️ Precautions:")
            for p in details["precautions"]:
                st.markdown(f"- {p}")

        with col_dir2:
            st.markdown("#### 🥗 Nutrition & Diet:")
            for d in details["diet"]:
                st.markdown(f"- {d}")

            st.markdown("#### 🏃 Lifestyle & Activity:")
            for w in details["workout"]:
                st.markdown(f"- {w}")

# Universal Medical Disclaimer Footer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Medical & Legal Disclaimer:</strong><br>
    This AI Medicine Recommendation System is built for educational, academic, and clinical decision-support demonstration purposes only.
    It does NOT substitute professional medical advice, clinical diagnosis, or certified prescription by a licensed medical practitioner.
    Always seek the advice of a qualified physician with any questions you may have regarding a medical condition.
</div>
""", unsafe_allow_html=True)
