# 🏥 MedPulse AI - Intelligent Medicine & Health Recommendation System

> **A Machine Learning-Powered Clinical Decision-Support & Pharmacotherapy Recommendation System**  
> Diagnoses 41 medical conditions from 132 symptoms, recommends first-line medications, precautions, dietary plans, and workouts, flags dangerous drug-drug interactions, finds affordable generic substitutes, and generates patient consultation summary reports.

---

## 🌟 Key Features

1. **🩺 Symptom-based Disease & Medication Recommendation**:
   - Predicts 41 different medical conditions with high accuracy based on 132 standardized symptoms.
   - Outputs primary condition, **differential diagnoses** (top 3 probabilities), and detailed medical description.
   - Recommends first-line medications, dosage advice, dietary guidelines, physical exercises, and clinical precautions.

2. **🧠 Smart Natural Language Symptom Extractor (NLP)**:
   - Users can describe their symptoms in free English text (e.g. *"I have severe headache, fever, shivering and vomiting"*).
   - Automated NLP entity extractor identifies matching clinical symptoms automatically.

3. **🚨 Emergency Triage & Red-Flag Symptom Detection**:
   - Detects critical conditions requiring urgent hospitalization (e.g., chest pain, hemoptysis, sudden unilateral weakness, acute abdominal bleeding) and immediately flashes clinical triage warnings.

4. **💊 Medicine Catalog & Generic Drug Substitute Finder**:
   - Search medicines by brand, active salt molecule, or indication.
   - Highlights **cheaper generic alternatives** to help patients save money on prescription costs.
   - Displays therapeutic class, standard dosage, and common side effects.

5. **⚠️ Pharmacological Drug-Drug Interaction Safety Checker**:
   - Evaluates combinations of 2 or more medications.
   - Detects severe, moderate, or minor drug interactions (e.g., Aspirin + NSAIDs, Metformin + Alcohol, Telmisartan + Potassium-sparing diuretics), clinical mechanisms, and physician action advice.

6. **📄 Downloadable Patient Consultation Report**:
   - Generates and exports a clean clinical consultation summary report (`.txt` / `.md`) with patient details, symptoms, predicted condition, medicines, and lifestyle guidelines.

7. **⚡ Dual Interface (Web Dashboard + REST API)**:
   - Interactive UI built with **Streamlit**.
   - Production-ready RESTful API built with **FastAPI** with auto-generated Swagger UI (`/docs`).

---

## 🏗️ Project Architecture

```
Medicine-Recommendation-System/
├── app.py                      # Main Streamlit Interactive Dashboard
├── api.py                      # FastAPI REST API Server
├── train_model.py              # ML Training & Evaluation Script
├── generate_dataset.py         # Medical Data & Auxiliary Database Generator
├── test_system.py              # Automated Unit & Integration Test Suite
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Documentation & Viva Guide
├── data/                       # Medical Datasets
│   ├── Training.csv            # 4,920 Training Records
│   ├── Testing.csv             # 410 Testing Records
│   ├── descriptions.csv        # Detailed Disease Overviews
│   ├── precautions.csv         # 4 Precautions per Condition
│   ├── medications.csv         # Standard Prescriptions & Dosages
│   ├── diets.csv               # Dietary Recommendations
│   ├── workouts.csv            # Physical Activity & Rest Plans
│   ├── medicine_catalog.csv    # Drug database & generic substitutes
│   └── drug_interactions.csv   # Drug-drug interaction rules
├── models/                     # Saved Model Artifacts
│   ├── best_model.pkl          # Trained Random Forest Model
│   ├── label_encoder.pkl       # Target Disease Label Encoder
│   ├── symptom_features.json   # 131 Standardized Symptom Feature Names
│   └── model_metrics.json      # Benchmark Metrics & Evaluation Results
└── src/                        # Core Application Engine
    ├── recommender.py          # Prediction Engine & Safety Rules
    ├── nlp_helper.py           # NLP Symptom Entity Extractor
    └── report_generator.py     # Patient Consultation Report Builder
```

---

## 📊 Machine Learning Benchmark

Four machine learning architectures were trained and benchmarked across 5-fold cross-validation:

| Model Architecture | Test Accuracy | 5-Fold CV Accuracy | Weighted F1-Score |
| :--- | :---: | :---: | :---: |
| **Random Forest (Selected)** | **100.0%** | **99.51%** | **1.00** |
| Support Vector Machine (SVC) | 100.0% | 99.55% | 1.00 |
| Multinomial Naive Bayes | 100.0% | 99.57% | 1.00 |
| Decision Tree Classifier | 100.0% | 96.63% | 1.00 |

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure Python 3.9+ is installed on your machine.

### 2. Navigate to Project Directory
```bash
cd "Medicine-Recommendation-System"
```

### 3. (Optional) Re-train the Model
The model is already pre-trained and saved in `models/`. To re-train from scratch:
```bash
python train_model.py
```

### 4. Run the Automated Test Suite
Verify that all unit and integration tests pass:
```bash
python test_system.py
```

### 5. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 6. Launch the FastAPI REST Server (Optional)
```bash
python -m uvicorn api:app --reload --port 8000
```
Open interactive Swagger API Documentation at `http://127.0.0.1:8000/docs`.

---

## 🌐 REST API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and overview |
| `GET` | `/symptoms` | List of all 132 detectable clinical symptoms |
| `POST` | `/predict` | Diagnose disease and get complete medications & precautions |
| `POST` | `/extract-symptoms` | Extract symptom keys from free text |
| `GET` | `/medicines/search` | Search medicine catalog and generic alternatives |
| `POST` | `/check-interactions` | Check drug-drug interaction safety between 2+ drugs |

### Example API Request (Predict Disease):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "symptoms": ["chills", "high_fever", "vomiting", "sweating", "headache"],
    "patient_name": "John Doe",
    "patient_age": 32,
    "patient_gender": "Male"
  }'
```

---

## 🎓 College / Viva Questions & Answers

**Q1: What is the main objective of this project?**  
> *Answer*: To provide an automated, AI-driven healthcare decision-support system that predicts diseases from reported symptoms, recommends appropriate medications, diet, and lifestyle precautions, identifies dangerous drug interactions, and suggests affordable generic substitutes.

**Q2: Which machine learning algorithms were used and why?**  
> *Answer*: We evaluated Random Forest, Support Vector Machines (Linear SVC), Multinomial Naive Bayes, and Decision Trees. Random Forest was selected as the champion model because of its high ensemble generalization, robustness against noise, resistance to overfitting, and ability to output probability distributions across multiple classes.

**Q3: How does the system handle patient symptom input in natural language?**  
> *Answer*: An NLP extraction module (`nlp_helper.py`) utilizes regex phrase matching and a curated clinical synonym ontology to map informal user complaints (e.g., *"tummy ache"*, *"loose motions"*, *"shaking"*) to standardized clinical feature vectors.

**Q4: How does the system ensure patient safety?**  
> *Answer*: Through three layers:
> 1. **Emergency Triage Module**: Instantly detects red-flag symptoms (chest pain, unilateral paralysis, hemoptysis) and alerts the patient to seek urgent emergency care.
> 2. **Drug-Drug Interaction Checker**: Checks if prescribed or taken medicines interact adversely.
> 3. **Medical Disclaimer**: Clearly indicates that the system is an educational decision-support tool and not a substitute for a licensed doctor.

---

## ⚠️ Medical & Legal Disclaimer
*This system is developed strictly for educational, academic, and research purposes. It is designed to assist and demonstrate clinical decision support concepts. It does not replace professional medical judgment, diagnosis, or treatment. Always seek the advice of a qualified physician with any health questions.*
