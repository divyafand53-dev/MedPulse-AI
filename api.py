"""
FastAPI REST API Server for Medicine Recommendation System.
Provides RESTful endpoints for disease prediction, symptom extraction,
medicine search, and drug interaction checks.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from src.recommender import MedicineRecommender
from src.nlp_helper import extract_symptoms_from_text

app = FastAPI(
    title="MedPulse AI - Medicine Recommendation System API",
    description="REST API for automated medical symptom diagnosis, drug recommendations, and interaction checks.",
    version="1.0.0"
)

# Initialize recommender
recommender = MedicineRecommender()
all_symptoms = recommender.get_all_symptoms()

# Pydantic Schemas
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str] = Field(..., example=["headache", "high_fever", "chills"])
    patient_name: Optional[str] = "Anonymous"
    patient_age: Optional[int] = 30
    patient_gender: Optional[str] = "Unspecified"

class NaturalLanguageRequest(BaseModel):
    text: str = Field(..., example="I have severe stomach pain, acidity and vomiting since yesterday.")

class InteractionCheckRequest(BaseModel):
    medicines: List[str] = Field(..., example=["Aspirin", "Ibuprofen / Aceclofenac"])

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "MedPulse AI Medicine Recommendation Engine",
        "supported_conditions": 41,
        "supported_symptoms": len(all_symptoms),
        "docs_url": "/docs"
    }

@app.get("/symptoms")
def get_symptoms():
    """Retrieve all 132 detectable clinical symptoms."""
    return {"total_symptoms": len(all_symptoms), "symptoms": all_symptoms}

@app.post("/predict")
def predict_disease(request: SymptomPredictionRequest):
    """Diagnose condition and prescribe medications based on symptom list."""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom must be provided.")

    result = recommender.predict_disease(request.symptoms)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "patient": {
            "name": request.patient_name,
            "age": request.patient_age,
            "gender": request.patient_gender
        },
        "diagnosis": result
    }

@app.post("/extract-symptoms")
def extract_symptoms(request: NaturalLanguageRequest):
    """Extract standard symptom keys from free-form natural language text."""
    detected = extract_symptoms_from_text(request.text, all_symptoms)
    return {
        "input_text": request.text,
        "detected_count": len(detected),
        "detected_symptoms": detected
    }

@app.get("/medicines/search")
def search_medicines(query: str = Query("", description="Brand name, active salt, category, or condition")):
    """Search medicine database for indications, side effects, and generic alternatives."""
    results = recommender.search_medicines(query)
    return {"total_results": len(results), "medicines": results}

@app.post("/check-interactions")
def check_interactions(request: InteractionCheckRequest):
    """Evaluate dangerous drug-drug interactions between two or more drugs."""
    if len(request.medicines) < 2:
        raise HTTPException(status_code=400, detail="At least two medicines are required for interaction checking.")

    interactions = recommender.check_drug_interactions(request.medicines)
    return {
        "medicines_checked": request.medicines,
        "has_interactions": len(interactions) > 0,
        "interaction_count": len(interactions),
        "interactions": interactions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
