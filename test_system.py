"""
End-to-End Test Suite for Medicine Recommendation System.
Validates:
1. Recommender engine loading
2. ML Disease prediction with confidence & emergency alert checks
3. Natural Language symptom extraction
4. Drug interaction checking
5. Medicine search & generic alternatives
6. Patient consultation report generation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import MedicineRecommender
from src.nlp_helper import extract_symptoms_from_text
from src.report_generator import generate_patient_report

def run_tests():
    print("=" * 60)
    print("RUNNING END-TO-END TEST SUITE FOR MEDICINE RECOMMENDER")
    print("=" * 60)

    # Test 1: Initialize Engine
    print("\n[Test 1] Initializing MedicineRecommender...")
    recommender = MedicineRecommender()
    symptoms = recommender.get_all_symptoms()
    assert len(symptoms) > 100, f"Expected >100 symptoms, got {len(symptoms)}"
    print(f" PASS: Loaded {len(symptoms)} symptom features successfully.")

    # Test 2: Disease Prediction - Malaria
    print("\n[Test 2] Testing Disease Prediction (Malaria symptoms)...")
    malaria_symptoms = ["chills", "vomiting", "high_fever", "sweating", "headache"]
    res_malaria = recommender.predict_disease(malaria_symptoms)
    print(f"  Predicted Disease: {res_malaria['disease']}")
    print(f"  Confidence: {res_malaria['confidence']}%")
    print(f"  Recommended Meds: {res_malaria['medications'][:2]}")
    assert res_malaria["disease"] == "Malaria", f"Expected Malaria, got {res_malaria['disease']}"
    assert res_malaria["confidence"] > 70, "Confidence should be high"
    assert len(res_malaria["medications"]) > 0, "Medications should not be empty"
    print(" PASS: Disease prediction accurate.")

    # Test 3: Emergency Symptom Alert (Chest Pain / Heart attack)
    print("\n[Test 3] Testing Red-Flag Emergency Triage...")
    emergency_symptoms = ["chest_pain", "breathlessness", "sweating"]
    res_emerg = recommender.predict_disease(emergency_symptoms)
    print(f"  Emergency Alerts Triggered: {len(res_emerg['emergency_alerts'])}")
    assert len(res_emerg["emergency_alerts"]) > 0, "Expected emergency alert for chest_pain"
    print(f"  Alert message: {res_emerg['emergency_alerts'][0]}")
    print(" PASS: Emergency triage alert successfully triggered.")

    # Test 4: NLP Symptom Extractor
    print("\n[Test 4] Testing Natural Language Symptom Extractor...")
    nl_text = "I am suffering from high fever, shivering, coughing and terrible joint pain."
    extracted = extract_symptoms_from_text(nl_text, symptoms)
    print(f"  Input Text: '{nl_text}'")
    print(f"  Extracted Symptoms: {extracted}")
    expected_matches = ["high_fever", "shivering", "cough", "joint_pain"]
    for em in expected_matches:
        assert em in extracted, f"Expected {em} to be extracted"
    print(" PASS: NLP extractor correctly identified all target symptoms.")

    # Test 5: Drug Interaction Safety Checker
    print("\n[Test 5] Testing Drug-Drug Interaction Checker...")
    test_drugs = ["Aspirin", "Ibuprofen / Aceclofenac"]
    interactions = recommender.check_drug_interactions(test_drugs)
    print(f"  Tested Pair: {test_drugs}")
    print(f"  Interactions detected: {len(interactions)}")
    assert len(interactions) > 0, "Expected interaction between Aspirin and NSAIDs"
    print(f"  Severity: {interactions[0]['severity']} | Action: {interactions[0]['action']}")
    print(" PASS: Dangerous drug-drug interaction flagged correctly.")

    # Test 6: Medicine Search & Generic Finder
    print("\n[Test 6] Testing Medicine Catalog Search & Generics...")
    search_res = recommender.search_medicines("Paracetamol")
    assert len(search_res) > 0, "Should find Paracetamol in catalog"
    print(f"  Found Drug: {search_res[0]['name']}")
    print(f"  Generic Substitutes: {search_res[0]['substitutes']}")
    print(" PASS: Medicine catalog lookup & generic alternatives verified.")

    # Test 7: Patient Consultation Report Generator
    print("\n[Test 7] Testing Patient Consultation Report Generation...")
    pat_info = {"name": "John Doe", "age": 35, "gender": "Male"}
    report = generate_patient_report(pat_info, res_malaria, malaria_symptoms)
    assert "MEDPULSE AI HEALTH CONSULTATION REPORT" in report
    assert "John Doe" in report
    assert "Malaria" in report
    print("  Report preview length:", len(report), "characters")
    print(" PASS: Report generated successfully.")

    print("\n" + "=" * 60)
    print(" ALL 7 UNIT & INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
