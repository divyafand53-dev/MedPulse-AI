"""
Dataset Generator for Medicine Recommendation System.
Generates comprehensive clinical data for:
- Symptoms & Disease mapping (Training & Testing datasets)
- Disease Descriptions
- Precautions
- Medications
- Diets & Nutrition
- Workouts & Lifestyle
- Medicine Catalog (with generic alternatives & side effects)
- Drug Interactions
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Define Symptoms (132 symptoms)
SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills",
    "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting",
    "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy",
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin",
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes",
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes",
    "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes",
    "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus",
    "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremities", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
    "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
    "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze"
]
# Remove duplicates if any while maintaining order
SYMPTOMS = list(dict.fromkeys(SYMPTOMS))

# 2. Disease Profile definitions (Primary symptoms, Secondary symptoms, Descriptions, Precautions, Medications, Diets, Workouts)
DISEASE_DATA = {
    "Fungal infection": {
        "primary": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        "secondary": ["skin_peeling", "burning_micturition"],
        "description": "Fungal infection is a common skin condition caused by fungi like dermatophytes or yeast, leading to itchy, red, circular or patchy rashes.",
        "precautions": ["Keep the affected skin clean and dry", "Use antifungal dusting powder or creams", "Avoid sharing towels, clothes, or personal items", "Wear loose, breathable cotton clothing"],
        "medications": ["Fluconazole 150mg (once weekly)", "Clotrimazole Topical Cream 1%", "Terbinafine 250mg tab", "Ketoconazole 2% Shampoo/Soap"],
        "diet": ["Probiotic-rich yogurt and kefir", "Garlic and turmeric in food", "Limit high-sugar and fermented foods", "Hydrate with plenty of clean water"],
        "workout": ["Light indoor stretching", "Avoid excessive sweating in damp clothes", "Shower immediately after exercising", "Practice yoga and relaxation techniques"]
    },
    "Allergy": {
        "primary": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
        "secondary": ["runny_nose", "redness_of_eyes", "throat_irritation", "itching"],
        "description": "An allergic reaction occurs when the immune system overreacts to foreign substances like pollen, pet dander, dust, or specific foods.",
        "precautions": ["Identify and avoid known allergen triggers", "Keep indoor air filtered using HEPA filters", "Wash hands and face after outdoor exposure", "Consult an allergist if symptoms persist"],
        "medications": ["Cetirizine 10mg tab", "Levocetirizine + Montelukast tab", "Fexofenadine 120mg tab", "Fluticasone nasal spray"],
        "diet": ["Warm herbal ginger and tulsi tea", "Vitamin C rich citrus fruits and berries", "Avoid known food allergens (nuts, shellfish)", "Include anti-inflammatory leafy greens"],
        "workout": ["Mild indoor walking", "Deep breathing exercises (Pranayama)", "Avoid exercising outdoors during high pollen hours", "Gentle stretching and mobility work"]
    },
    "GERD": {
        "primary": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "chest_pain"],
        "secondary": ["cough", "throat_irritation", "indigestion", "loss_of_appetite"],
        "description": "Gastroesophageal Reflux Disease (GERD) is a chronic digestive condition where stomach acid or bile irritates the food pipe lining.",
        "precautions": ["Eat smaller, more frequent meals", "Avoid lying down for at least 2 hours after eating", "Elevate head of bed by 6 inches", "Avoid spicy, greasy, and acidic foods"],
        "medications": ["Pantoprazole 40mg (empty stomach)", "Omeprazole 20mg tab", "Antacid Gel (Aluminium/Magnesium Hydroxide)", "Domperidone 10mg tab"],
        "diet": ["Oatmeal, non-citrus fruits like bananas and melons", "Steamed vegetables and lean proteins", "Avoid coffee, alcohol, carbonated soda, and chocolate", "Drink coconut water and chamomile tea"],
        "workout": ["Brisk walking after meals", "Upright posture yoga asanas", "Avoid intense core crunches or inverted poses", "Light cycling or swimming"]
    },
    "Chronic cholestasis": {
        "primary": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain"],
        "secondary": ["yellowing_of_eyes", "dark_urine", "fatigue"],
        "description": "Chronic cholestasis is a long-standing impairment of bile flow from the liver to the duodenum, leading to bile accumulation in the liver.",
        "precautions": ["Limit dietary saturated fats", "Avoid alcohol and hepatotoxic drugs completely", "Take fat-soluble vitamin supplements (A, D, E, K)", "Regular liver function test monitoring"],
        "medications": ["Ursodeoxycholic Acid (UDCA) 300mg", "Cholestyramine resin", "Fat-soluble vitamin supplements", "Hydroxyzine (for severe pruritus)"],
        "diet": ["Low-fat diet with medium-chain triglycerides (MCT)", "High fiber oats, barley, and fresh vegetables", "Avoid fried and heavy oily meals", "Lean poultry and fish in moderation"],
        "workout": ["Gentle daily walking", "Low-impact aerobic conditioning", "Adequate rest periods between activities", "Avoid high-exhaustion resistance training"]
    },
    "Drug Reaction": {
        "primary": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_urination"],
        "secondary": ["blister", "vomiting", "nausea", "headache"],
        "description": "An adverse drug reaction is an unwanted or harmful reaction experienced following the administration of a drug or combination of drugs.",
        "precautions": ["Immediately discontinue the suspected medication", "Consult a physician or visit an emergency room", "Keep a medical alert card listing drug allergies", "Stay hydrated with clean water and ORS"],
        "medications": ["Hydroxyzine 25mg tab", "Prednisolone 20mg (short doctor course)", "Calamine lotion for skin soothing", "Oral rehydration solution (ORS)"],
        "diet": ["Bland diet: boiled rice, toast, clear broths", "Ample water and electrolyte hydration", "Avoid processed foods, dairy, and artificial additives", "Fresh coconut water"],
        "workout": ["Strict physical rest until symptoms resolve", "Avoid sweating which irritates rash", "Gentle deep breathing in well-ventilated room", "Light seated stretching if comfortable"]
    },
    "Peptic ulcer disease": {
        "primary": ["vomiting", "loss_of_appetite", "abdominal_pain", "passage_of_gases", "internal_itching"],
        "secondary": ["stomach_pain", "indigestion", "acidity"],
        "description": "Peptic ulcer disease involves open sores that develop on the inside lining of your stomach and the upper portion of your small intestine.",
        "precautions": ["Avoid NSAID pain killers (like Ibuprofen/Aspirin)", "Eat meals at consistent, regular times", "Avoid smoking, alcohol, and caffeine", "Manage psychological stress effectively"],
        "medications": ["Rabeprazole 20mg + Amoxicillin/Clarithromycin (if H. pylori)", "Sucralfate 1g oral suspension", "Pantoprazole 40mg tab", "Ranitidine/Famotidine 20mg"],
        "diet": ["Probiotic foods like curd and kefir", "High-fiber apples, pears, and carrots", "Avoid chili, black pepper, and citrus juices", "Cabbage juice and chamomile tea"],
        "workout": ["Low-intensity morning walking", "Gentle yoga and meditation for stress reduction", "Avoid heavy weightlifting that increases intra-abdominal pressure", "Regular breathing exercises"]
    },
    "AIDS": {
        "primary": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
        "secondary": ["fatigue", "weight_loss", "swelled_lymph_nodes", "diarrhoea"],
        "description": "Acquired Immunodeficiency Syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the Human Immunodeficiency Virus (HIV).",
        "precautions": ["Strict adherence to Antiretroviral Therapy (ART)", "Practice safe protected intercourse", "Prevent secondary opportunistic infections", "Consume strictly sanitized, cooked food"],
        "medications": ["Tenofovir + Lamivudine + Dolutegravir (TLD regimen)", "Cotrimoxazole (prophylaxis)", "Multivitamin and mineral supplements", "Antifungal nystatin oral suspension"],
        "diet": ["Calorie-dense, high-protein balanced meals (eggs, legumes, fish)", "Pasteurized milk and well-cooked foods only", "Clean filtered or boiled drinking water", "Nutrient-rich fruit smoothies"],
        "workout": ["Gradual resistance training to prevent muscle wasting", "Moderate walking 20-30 minutes daily", "Adequate sleep and restorative rest", "Flexibility and balance exercises"]
    },
    "Diabetes": {
        "primary": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "excessive_hunger", "polyuria"],
        "secondary": ["increased_appetite", "weight_gain", "obesity"],
        "description": "Diabetes mellitus is a metabolic disorder characterized by high blood glucose levels over a prolonged period due to insulin deficiency or resistance.",
        "precautions": ["Monitor blood glucose levels regularly", "Maintain consistent meal timings and carbohydrate counts", "Perform daily foot inspections for sores or cuts", "Annual screening for retinopathy and nephropathy"],
        "medications": ["Metformin 500mg/1000mg tab", "Glimepiride 1mg/2mg tab", "Vildagliptin 50mg tab", "Human Insulin (as prescribed by endocrinologist)"],
        "diet": ["Low glycemic index foods (whole oats, quinoa, lentils)", "Abundant leafy green vegetables and bitter gourd", "Strictly eliminate refined sugar, candies, and sweetened beverages", "Nuts like almonds and walnuts for healthy fats"],
        "workout": ["30-45 minutes brisk walking daily", "Moderate strength training 3 times weekly", "Post-meal 10-15 minute walk to lower glucose spike", "Yoga postures like Dhanurasana and Paschimottanasana"]
    },
    "Gastroenteritis": {
        "primary": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
        "secondary": ["abdominal_pain", "mild_fever", "nausea"],
        "description": "Gastroenteritis (stomach flu) is an intestinal infection marked by watery diarrhoea, abdominal cramps, nausea or vomiting, and sometimes fever.",
        "precautions": ["Replenish lost fluids and electrolytes aggressively", "Maintain strict hand hygiene before meals", "Avoid dairy products and fatty foods during active infection", "Rest adequately to allow bowel recovery"],
        "medications": ["Oral Rehydration Salts (ORS) solution", "Ondansetron 4mg tab (for nausea/vomiting)", "Zinc supplements 20mg tab", "Racecadotril 100mg cap"],
        "diet": ["BRAT diet (Bananas, Rice, Applesauce, Toast)", "Electrolyte fluids, coconut water, salted rice congee", "Avoid spicy food, caffeine, and dairy products", "Clear vegetable broths"],
        "workout": ["Strict bed rest during acute phase", "Avoid all strenuous physical exertion", "Short slow indoor walks once hydrated", "Full recovery before resuming workouts"]
    },
    "Bronchial Asthma": {
        "primary": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
        "secondary": ["chest_pain", "fast_heart_rate"],
        "description": "Bronchial asthma is a chronic respiratory condition in which the airways narrow, swell, and produce extra mucus, causing difficulty in breathing.",
        "precautions": ["Always carry a quick-relief rescue inhaler", "Avoid exposure to smoke, dust, cold air, and pet dander", "Take prescribed controller inhalers consistently", "Monitor peak expiratory flow rate (PEFR)"],
        "medications": ["Salbutamol / Albuterol Inhaler 100mcg (rescue)", "Budesonide + Formoterol DPI/MDI (maintenance)", "Montelukast 10mg tab", "Ipratropium bromide nebules"],
        "diet": ["Warm herbal ginger, peppermint, and eucalyptus teas", "Foods rich in magnesium and omega-3 fatty acids", "Avoid sulfites found in dried fruits and processed wines", "Fresh fruits rich in vitamin C"],
        "workout": ["Warm-up thoroughly before any exercise", "Pursed-lip and diaphragmatic breathing drills", "Indoor swimming in non-chlorinated warm water", "Gentle walking on mild, warm weather days"]
    },
    "Hypertension": {
        "primary": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
        "secondary": ["fast_heart_rate", "palpitations"],
        "description": "Hypertension (high blood pressure) is a common condition where the long-term force of blood against artery walls is elevated enough to cause health issues.",
        "precautions": ["Reduce daily sodium/salt intake below 2,000 mg", "Monitor blood pressure regularly at home", "Quit smoking and moderate/avoid alcohol", "Manage daily emotional and mental stress"],
        "medications": ["Amlodipine 5mg tab", "Telmisartan 40mg tab", "Hydrochlorothiazide 12.5mg tab", "Atenolol 25mg tab"],
        "diet": ["DASH diet rich in vegetables, fruits, and low-fat dairy", "Potassium-rich foods: bananas, spinach, sweet potatoes", "Strictly restrict table salt, pickles, and processed snacks", "Garlic and flaxseeds in daily diet"],
        "workout": ["30 minutes of aerobic exercise (brisk walking, cycling) daily", "Gentle swimming or water aerobics", "Avoid heavy isometric bench pressing or breath holding", "Pranayama and guided meditation for 15 minutes"]
    },
    "Migraine": {
        "primary": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability", "visual_disturbances"],
        "secondary": ["vomiting", "nausea"],
        "description": "Migraine is a neurological condition that causes intense, debilitating throbbing headache typically on one side of the head, often with aura.",
        "precautions": ["Rest in a dark, quiet, noise-free room during attacks", "Maintain a regular sleep schedule and sleep duration", "Keep a headache diary to identify trigger foods", "Stay adequately hydrated throughout the day"],
        "medications": ["Sumatriptan 50mg/100mg tab", "Naproxen 500mg tab", "Paracetamol 650mg + Domperidone 10mg", "Propranolol 40mg (preventative)"],
        "diet": ["Magnesium-rich foods (almonds, pumpkin seeds, spinach)", "Regular balanced meals; avoid skipping meals or fasting", "Avoid aged cheeses, MSG, chocolate, red wine, and aspartame", "Adequate water and herbal chamomile tea"],
        "workout": ["Gentle neck and shoulder stretching exercises", "Yoga nidra and progressive muscle relaxation", "Avoid intense high-impact cardio during migraine spells", "Regular calm morning walking"]
    },
    "Cervical spondylosis": {
        "primary": ["back_pain", "neck_pain", "dizziness", "loss_of_balance"],
        "secondary": ["weakness_in_limbs"],
        "description": "Cervical spondylosis is an age-related wear and tear of the spinal discs and joints in the neck leading to stiffness, chronic pain, and nerve pinching.",
        "precautions": ["Maintain ergonomic workstation posture with eye-level screen", "Use a supportive cervical contour pillow", "Avoid carrying heavy backpacks or sudden neck jerks", "Take regular micro-breaks during prolonged desk work"],
        "medications": ["Aceclofenac 100mg + Paracetamol 325mg tab", "Thiocolchicoside 4mg (muscle relaxant)", "Pregabalin 75mg (for neuropathic shooting pain)", "Methylcobalamin (B12) 1500mcg"],
        "diet": ["Calcium-rich dairy, sesame seeds, and fortified plant milks", "Vitamin D rich foods and sunlight exposure", "Anti-inflammatory foods: ginger, berries, and omega-3s", "Avoid ultra-processed inflammatory junk foods"],
        "workout": ["Isometric cervical neck strengthening exercises", "Gentle chin tucks and shoulder blade retractions", "Upper back mobility stretches", "Avoid high-impact running or contact sports"]
    },
    "Paralysis (brain hemorrhage)": {
        "primary": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
        "secondary": ["slurred_speech", "loss_of_balance"],
        "description": "Hemiplegia or paralysis following a cerebrovascular brain hemorrhage requires immediate emergency neurological intervention and intensive rehabilitation.",
        "precautions": ["Immediate emergency hospitalization (dial emergency 108/911)", "Strict blood pressure control", "Prevent pressure ulcers/bedsores through regular patient turning", "Begin early supervised physical and occupational therapy"],
        "medications": ["Mannitol 20% IV (under hospital supervision)", "Labetalol / Nicardipine (blood pressure control)", "Citicoline 500mg tab (neuroprotective)", "Physiotherapy support medications"],
        "diet": ["Pureed or soft swallow-safe foods as approved by speech therapist", "Low-sodium, heart-healthy nutrient-dense meals", "High-fiber diet to prevent constipation and straining", "Adequate hydration via assisted feeding"],
        "workout": ["Professional neuro-physiotherapy and range of motion (ROM)", "Passive joint mobilization to prevent joint contractures", "Assisted sit-to-stand training as recovery progresses", "Fine motor occupational therapy exercises"]
    },
    "Jaundice": {
        "primary": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
        "secondary": ["loss_of_appetite", "yellowing_of_eyes"],
        "description": "Jaundice is a medical condition caused by elevated levels of bilirubin in the bloodstream, resulting in yellow pigmentation of the skin, sclera, and mucous membranes.",
        "precautions": ["Take complete bed rest during the acute jaundice phase", "Drink strictly boiled or filtered water", "Avoid oily, greasy, and spicy foods completely", "Get regular liver function tests (LFT) to monitor bilirubin"],
        "medications": ["Liv-52 DS tablet / Silymarin 140mg tab", "Ursodeoxycholic Acid 300mg tab", "Vitamin B-complex with L-ornithine L-aspartate", "ORS fluids for hydration"],
        "diet": ["Sugarcane juice (fresh and hygienically prepared)", "Boiled rice with light moong dal soup", "Papaya, sweet lime, and tender coconut water", "Avoid alcohol, red meat, butter, and deep-fried items completely"],
        "workout": ["Strict physical bed rest until bilirubin levels normalize", "Avoid any vigorous exercise or heavy household chores", "Slow indoor walking once energy begins to return", "Gentle restorative breathing while seated"]
    },
    "Malaria": {
        "primary": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        "secondary": ["diarrhoea", "loss_of_appetite"],
        "description": "Malaria is a mosquito-borne infectious disease caused by Plasmodium parasites transmitted to humans through the bites of infected female Anopheles mosquitoes.",
        "precautions": ["Use insecticide-treated bed nets and insect repellents", "Eliminate stagnant water around residential spaces", "Complete the full course of prescribed antimalarials", "Wear long-sleeved clothing during dusk and dawn"],
        "medications": ["Artemether + Lumefantrine tab (ACT)", "Chloroquine phosphate 250mg tab", "Primaquine 15mg tab (for radical cure as per species)", "Paracetamol 650mg for high fever"],
        "diet": ["High-energy carbohydrate soups and porridge", "Electrolyte fluids, coconut water, and citrus juice", "Easily digestible steamed foods: khichdi, idli", "Avoid heavy oils and spicy rich gravies"],
        "workout": ["Complete bed rest until fever subsides completely", "Hydration and temperature monitoring", "Resume normal walking only after full parasitemia clearance", "Light stretching during late convalescence"]
    },
    "Chicken pox": {
        "primary": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
        "secondary": ["blister"],
        "description": "Chickenpox is a highly contagious viral infection caused by the Varicella-Zoster virus, marked by an itchy, blistering skin rash and constitutional flu symptoms.",
        "precautions": ["Isolate the patient to prevent spreading to non-immune individuals", "Do not scratch blisters to prevent secondary bacterial infection and scarring", "Trim fingernails short and keep skin clean", "Use cool compresses or oatmeal baths to soothe itching"],
        "medications": ["Acyclovir 800mg (5 times daily for 7 days)", "Calamine lotion topically on itchy lesions", "Paracetamol 500mg for fever (Avoid Aspirin in children!)", "Cetirizine 10mg tab for itching"],
        "diet": ["Soft, cool, and non-acidic foods (applesauce, mashed potatoes)", "Plenty of water, tender coconut water, and fruit juices", "Avoid salty, spicy, and citrus foods that irritate mouth sores", "Warm vegetable soups and broths"],
        "workout": ["Strict home isolation and bed rest", "Avoid sweating which aggravates rash irritation", "Light indoor activities like reading or gentle sitting", "No gym, sports, or school/work until all crusts dry"]
    },
    "Dengue": {
        "primary": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain", "red_spots_over_body"],
        "secondary": ["malaise", "sweating"],
        "description": "Dengue fever is a mosquito-borne tropical viral disease transmitted by Aedes mosquitoes, causing sudden high fever, retro-orbital eye pain, severe arthralgia, and risk of thrombocytopenia.",
        "precautions": ["Monitor platelet counts and hematocrit daily", "Watch for warning signs: abdominal pain, persistent vomiting, mucosal bleeding", "Use mosquito repellents and prevent stagnant water pooling", "Strictly avoid NSAIDs like Ibuprofen or Aspirin due to bleeding risk"],
        "medications": ["Paracetamol 650mg for fever and aches", "Carica Papaya leaf extract syrup / tablets", "Oral Rehydration Salts (ORS) continuously", "Intravenous normal saline fluids if hospitalized"],
        "diet": ["Carica papaya leaf juice / extract", "Coconut water, pomegranate juice, and kiwi fruit", "Light nutrient-rich soups and khichdi", "Avoid dark-colored foods that could mask gastrointestinal bleeding"],
        "workout": ["Strict complete bed rest during the febrile and critical phase", "Avoid any bumps or falls that could trigger internal bleeding", "Gradual reconditioning over 2-3 weeks post-recovery", "Gentle walking only after platelet count stabilizes above 150,000"]
    },
    "Typhoid": {
        "primary": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_(typhos)", "belly_pain"],
        "secondary": ["loss_of_appetite"],
        "description": "Typhoid fever is a life-threatening systemic infection caused by the bacterium Salmonella enterica serotype Typhi, transmitted through contaminated food or water.",
        "precautions": ["Drink only boiled or certified bottled water", "Maintain meticulous hand hygiene with soap before eating", "Ensure all food is thoroughly cooked and served hot", "Complete the full antibiotic course to eliminate carrier status"],
        "medications": ["Azithromycin 500mg tab / Cefixime 200mg tab", "Ceftriaxone 1g/2g IV (for severe inpatient cases)", "Paracetamol 650mg for step-ladder fever", "Probiotics to restore gut flora"],
        "diet": ["High-calorie soft diet: boiled potatoes, porridge, banana", "Electrolyte water, strained barley water, and coconut water", "Avoid raw vegetables, unpeeled fruits, and street foods", "Avoid high-fiber foods that irritate inflamed intestinal Peyer's patches"],
        "workout": ["Bed rest until temperature stays normal for at least 48 hours", "Avoid any heavy lifting or strenuous exertion to prevent bowel perforation", "Slow short indoor walks during convalescence", "Full physical resumption only after physician clearance"]
    },
    "Hepatitis A": {
        "primary": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
        "secondary": ["fatigue"],
        "description": "Hepatitis A is an acute infectious disease of the liver caused by Hepatitis A virus (HAV), predominantly spread through the fecal-oral ingestion of contaminated water or food.",
        "precautions": ["Vaccinate family and contacts against Hepatitis A", "Drink boiled or mineral water exclusively", "Maintain strict hand hygiene after restroom use", "Avoid alcohol and unnecessary over-the-counter medications"],
        "medications": ["Supportive antiemetic: Ondansetron 4mg tab", "Silymarin / Ursodeoxycholic Acid 300mg", "B-complex vitamins with Zinc", "ORS and IV fluids if vomiting"],
        "diet": ["High-carbohydrate easily digestible diet (sugarcane juice, honey, rice)", "Fresh fruit juices (apple, watermelon)", "Strictly zero fried, greasy, or fatty foods", "Frequent small meals rather than heavy dinners"],
        "workout": ["Rest until liver enzymes (SGPT/SGOT) decline significantly", "Avoid strenuous physical activity or sport", "Light daily routine tasks once jaundice fades", "Gentle stretching and deep breathing"]
    },
    "Hepatitis B": {
        "primary": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
        "secondary": ["joint_pain", "nausea"],
        "description": "Hepatitis B is a serious liver infection caused by the Hepatitis B virus (HBV), transmitted via exposure to infectious blood, semen, or other body fluids.",
        "precautions": ["Get close contacts vaccinated with Hepatitis B vaccine series", "Never share razors, needles, toothbrushes, or manicuring tools", "Practice protected intercourse", "Regular ultrasound liver imaging and viral load monitoring"],
        "medications": ["Tenofovir Disoproxil Fumarate 300mg tab", "Entecavir 0.5mg tab", "Pegylated Interferon alfa-2a (in specialized cases)", "Liver protective antioxidants"],
        "diet": ["Balanced Mediterranean-style diet (vegetables, olive oil, beans)", "Lean proteins: tofu, egg whites, and skinless poultry", "Strict lifelong avoidance of alcoholic drinks", "Cruciferous vegetables (broccoli, cabbage) to support detoxification"],
        "workout": ["Moderate aerobic walking 30 minutes, 4 days a week", "Low-intensity resistance training", "Rest whenever experiencing episodic fatigue", "Stress reduction yoga and tai chi"]
    },
    "Hepatitis C": {
        "primary": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "family_history"],
        "secondary": ["dark_urine", "joint_pain"],
        "description": "Hepatitis C is a viral infection caused by HCV that attacks the liver, often leading to chronic inflammation, cirrhosis, and potential hepatocellular carcinoma.",
        "precautions": ["Avoid all alcohol and hepatotoxic compounds", "Screen family members and check HCV RNA viral load", "Do not share personal grooming or medical injection equipment", "Screen for liver fibrosis via FibroScan"],
        "medications": ["Sofosbuvir + Velpatasvir 400mg/100mg tab (Epclusa - 12 week cure)", "Ledipasvir + Sofosbuvir (Harvoni)", "Antioxidant multivitamin supplements"],
        "diet": ["Liver-friendly diet with high fiber and low sodium", "Green tea, walnuts, and berries rich in polyphenols", "Avoid iron-fortified supplements unless recommended by doctor", "Limit saturated animal fats"],
        "workout": ["Light to moderate regular walking", "Gentle resistance bands exercises", "Yoga for mental health and relaxation", "Avoid extreme high-exhaustion endurance sports"]
    },
    "Hepatitis D": {
        "primary": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
        "secondary": ["mild_fever"],
        "description": "Hepatitis D (Delta hepatitis) is a severe liver disease caused by the Hepatitis D virus, which can only propagate in the presence of Hepatitis B virus.",
        "precautions": ["Vaccination against Hepatitis B provides dual prevention", "Strict adherence to hepatologist supervision", "Avoid sharing personal hygiene items and needles", "Regular monitoring of liver fibrosis and AFP markers"],
        "medications": ["Bulevirtide (Hepcludex) entry inhibitor", "Pegylated Interferon Alfa-2a", "Tenofovir / Entecavir for concurrent HBV suppression"],
        "diet": ["Nutrient-dense, low-fat plant-forward meals", "Ample water and electrolyte fluids", "Zero alcohol consumption", "High-potassium fruits and steamed veggies"],
        "workout": ["Gentle mobility and restorative yoga", "Light evening walks", "Avoid physical fatigue and heavy weight training", "Restful 8-hour nightly sleep cycle"]
    },
    "Hepatitis E": {
        "primary": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure"],
        "secondary": ["mild_fever"],
        "description": "Hepatitis E is an enterically transmitted waterborne viral liver infection, particularly dangerous and potentially fatal for pregnant women.",
        "precautions": ["Ensure strict drinking water sanitation and boiling", "Avoid undercooked pork, venison, or shellfish", "Extra vigilant clinical monitoring in pregnant patients", "Proper sewage and community sanitation measures"],
        "medications": ["Supportive symptomatic therapy (Antiemetics, ORS)", "Ribavirin (in severe or chronic immunosuppressed cases)", "Vitamin B-complex with L-ornithine", "Hospitalization for acute monitoring"],
        "diet": ["Hydrating fluids, tender coconut water, diluted fruit juices", "Light boiled rice, steamed vegetables, and lentils", "Avoid all greasy, fried, and heavily salted meals", "Fresh ripe sweet fruits"],
        "workout": ["Complete rest until acute jaundice and liver enzymes subside", "Avoid any exercise that causes physical strain or sweating", "Slow walking indoors once appetite improves", "Full recovery before physical resumption"]
    },
    "Alcoholic hepatitis": {
        "primary": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
        "secondary": ["loss_of_appetite", "yellowing_of_eyes"],
        "description": "Alcoholic hepatitis is severe inflammation of the liver caused by prolonged and excessive heavy alcohol consumption, often accompanied by ascites and jaundice.",
        "precautions": ["Immediate, absolute cessation of alcohol consumption", "Enroll in addiction support and medical de-addiction programs", "Strict sodium-restricted diet (<2g/day) to control ascites", "Watch for signs of gastrointestinal bleeding or mental confusion"],
        "medications": ["Prednisolone 40mg/day (in severe non-infected cases with Maddrey DF >= 32)", "Spironolactone 100mg + Furosemide 40mg (diuretics for ascites)", "Thiamine (Vitamin B1) 100mg IV/oral", "Lactulose syrup (to prevent hepatic encephalopathy)"],
        "diet": ["High-protein, calorie-dense nutrition (1.5g protein/kg body weight)", "Severely restricted sodium/salt intake", "Frequent small meals including a late-evening carbohydrate snack", "Multivitamin supplementation with zinc and folate"],
        "workout": ["Mild seated physiotherapy exercises to combat muscle wasting", "Short gentle walks around home environment as tolerated", "Avoid strenuous physical exertion that strains ascites", "Deep breathing exercises"]
    },
    "Tuberculosis": {
        "primary": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "phlegm", "chest_pain", "blood_in_sputum"],
        "secondary": ["swelled_lymph_nodes", "malaise"],
        "description": "Tuberculosis (TB) is a contagious bacterial infection caused by Mycobacterium tuberculosis, primarily attacking the lungs and spread via respiratory droplets.",
        "precautions": ["Strictly finish the 6-month Directly Observed Therapy (DOTS) antibiotic regimen", "Wear an N95 mask to prevent infecting family members during initial weeks", "Ensure excellent room ventilation and sunlight in living spaces", "Monitor liver functions during anti-tubercular drug therapy"],
        "medications": ["Rifampicin 450/600mg tab", "Isoniazid 300mg tab", "Pyrazinamide 1500mg tab", "Ethambutol 800mg tab", "Pyridoxine (Vitamin B6) 50mg (to prevent neuropathy)"],
        "diet": ["High-protein diet: eggs, milk, lentils, paneer, and lean poultry", "Calorie-dense foods with healthy fats (ghee, nuts, seeds)", "Citrus fruits rich in Vitamin C, plus Vitamin A foods (carrots, spinach)", "Avoid alcohol and tobacco completely"],
        "workout": ["Strict rest during initial infectious and fever phase", "Deep breathing exercises (Incentive spirometry) to expand lung capacity", "Gradual slow walking as lung healing progresses", "Avoid exhausting cardio or lifting until sputum clears negative"]
    },
    "Common Cold": {
        "primary": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell", "muscle_pain"],
        "secondary": ["watering_from_eyes"],
        "description": "The common cold is a viral infectious disease of the upper respiratory tract, most commonly caused by Rhinoviruses, causing rhinorrhea, nasal congestion, and sore throat.",
        "precautions": ["Stay well-hydrated with warm fluids and broths", "Inhale warm steam with menthol or eucalyptus drops", "Wash hands regularly to curb transmission", "Rest adequately to allow the immune system to recover"],
        "medications": ["Paracetamol 500mg / Phenylephrine 10mg tab", "Cetirizine 10mg / Levocetirizine tab", "Saline nasal spray / drops", "Lozenges with honey and menthol"],
        "diet": ["Hot chicken soup or vegetable ginger soup", "Warm water with honey, lemon, and freshly grated ginger", "Vitamin C rich citrus fruits, kiwi, and bell peppers", "Avoid ice-cold beverages, dairy excess, and oily fried foods"],
        "workout": ["Mild restorative stretching at home", "Gentle indoor walking if 'neck-up' mild symptoms", "Avoid high-intensity workouts when running a fever", "Prioritize 8-9 hours of restorative sleep"]
    },
    "Pneumonia": {
        "primary": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
        "secondary": ["loss_of_appetite"],
        "description": "Pneumonia is an inflammatory condition of the lung primarily affecting the microscopic air sacs (alveoli), caused by bacterial or viral infections.",
        "precautions": ["Complete full prescribed antibiotic or antiviral regimen without missing doses", "Use a pulse oximeter to track blood oxygen saturation (SpO2 > 94%)", "Seek immediate emergency help if breathing becomes labored or blue lips occur", "Annual pneumococcal and influenza immunization"],
        "medications": ["Amoxicillin-Clavulanate 625mg tab / Azithromycin 500mg tab", "Levofloxacin 500mg tab (for atypical pneumonia)", "Paracetamol 650mg for high fever", "Mucolytic: N-Acetylcysteine 600mg effervescent tab"],
        "diet": ["Warm nutrient-dense broths and pureed vegetable soups", "Protein-rich scrambled eggs, lentils, and Greek yogurt", "Adequate warm water and electrolyte fluids", "Avoid dairy products if they cause throat mucus thickness"],
        "workout": ["Bed rest during the active febrile infection", "Incentive spirometry breathing device exercises 3 times daily", "Gradual slow walking once fever is gone for 72 hours", "Avoid cold outdoor air and heavy aerobic training for 3-4 weeks"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "primary": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        "secondary": ["stomach_pain"],
        "description": "Hemorrhoids (piles) are swollen veins in your lower rectum and anus, causing severe rectal pain, itching, and bright red bleeding during defecation.",
        "precautions": ["Take warm water sitz baths for 15-20 minutes after bowel movements", "Avoid straining or sitting on the toilet for prolonged periods", "Increase dietary soluble fiber to soften stool", "Stay well-hydrated by drinking at least 2.5 to 3 liters of water daily"],
        "medications": ["Calcium Dobesilate 500mg cap / Micronized Flavonoids 500mg tab", "Hydrocortisone + Lidocaine rectal ointment", "Psyllium husk (Isabgol) bulk laxative", "Stool softener: Docusate sodium / Lactulose syrup"],
        "diet": ["High-fiber foods: oats, chia seeds, figs, prunes, and beans", "At least 3 liters of water throughout the day", "Papaya, ripe bananas, and green leafy vegetables", "Strictly avoid spicy chilies, deep-fried snacks, and low-fiber white flour (maida)"],
        "workout": ["Kegel pelvic floor exercises to strengthen anal musculature", "30 minutes of brisk daily walking to stimulate gut peristalsis", "Avoid heavy squats, deadlifts, and straining breath holding", "Gentle hamstring and lower back stretching"]
    },
    "Heart attack": {
        "primary": ["vomiting", "breathlessness", "sweating", "chest_pain", "fast_heart_rate"],
        "secondary": ["palpitations", "anxiety", "dizziness"],
        "description": "Myocardial infarction (heart attack) occurs when blood flow decreases or stops to a part of the heart, causing damage to the heart muscle. Medical Emergency!",
        "precautions": ["IMMEDIATE EMERGENCY: Call 108 / 911 without delay", "Chew an uncoated Aspirin 300mg immediately if not contraindicated", "Have the patient sit down, remain calm, and loosen tight clothing", "Administer sublingual Nitroglycerin if previously prescribed by cardiologist"],
        "medications": ["Aspirin 300mg (chewed stat) + Clopidogrel 300mg stat", "Atorvastatin 80mg tab", "Sublingual Nitroglycerin 0.4mg spray/tab", "Emergency Hospital Cardiac Catheterization / Thrombolytic therapy"],
        "diet": ["Strictly low-sodium, low-cholesterol cardiac diet", "Whole grains, oats, extra virgin olive oil, and steamed greens", "Rich in omega-3: walnuts, flaxseeds, salmon", "Strict zero trans-fats, processed meats, and refined sugars"],
        "workout": ["Supervised phase-1 cardiac rehabilitation program only", "Slow paced walking under physician monitoring", "Strictly forbid heavy lifting, strenuous exertion, or competitive sports", "Relaxation deep breathing and guided mental tranquility"]
    },
    "Varicose veins": {
        "primary": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
        "secondary": ["painful_walking"],
        "description": "Varicose veins are twisted, enlarged, and engorged veins, most commonly appearing in the legs when vein valves become weak or damaged.",
        "precautions": ["Wear graduated compression stockings (Class II) during daytime", "Elevate legs above heart level for 15-20 minutes twice daily", "Avoid sitting or standing still for prolonged uninterrupted hours", "Maintain a healthy body weight to relieve venous pressure"],
        "medications": ["Micronized Purified Flavonoid Fraction (MPFF) 500mg tab", "Calcium dobesilate 500mg tab", "Horse Chestnut extract (Aescin)", "Topical heparin gel / soothing creams"],
        "diet": ["Flavonoid-rich berries, citrus fruits, and dark chocolate", "High-fiber diet to prevent chronic constipation (which worsens venous pressure)", "Potassium-rich foods (almonds, spinach, bananas) to reduce water retention", "Low-salt meals to minimize leg edema"],
        "workout": ["Low-impact walking 30-40 minutes daily to engage calf muscle pump", "Ankle pumps, leg lifts, and calf raises while seated or lying down", "Swimming and low-resistance stationary cycling", "Avoid high-impact jumping or prolonged heavy squatting"]
    },
    "Hypothyroidism": {
        "primary": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremities", "abnormal_menstruation"],
        "secondary": ["depression"],
        "description": "Hypothyroidism (underactive thyroid) is a condition where the thyroid gland does not produce enough crucial thyroid hormones, slowing body metabolism.",
        "precautions": ["Take thyroid medication first thing every morning on an empty stomach with plain water", "Wait at least 45 minutes to 1 hour before eating breakfast or drinking tea/coffee", "Have serum TSH and Free T4 tested every 6 to 12 months", "Avoid taking calcium or iron supplements within 4 hours of thyroid pill"],
        "medications": ["Levothyroxine Sodium 25mcg / 50mcg / 75mcg / 100mcg tab", "Selenium supplements 100mcg", "Vitamin D3 60,000 IU weekly (for common deficiency)", "Vitamin B12 supplements"],
        "diet": ["Iodized salt in moderate amounts", "Selenium and zinc-rich foods: pumpkin seeds, Brazil nuts, eggs", "Cook cruciferous vegetables (cabbage, broccoli) well before eating (steaming deactivates goitrogens)", "Avoid gluten or dairy if diagnosed with Hashimoto's thyroiditis"],
        "workout": ["Moderate aerobic workouts (fast walking, light jogging) 4 times a week", "Strength and resistance training to boost resting metabolic rate", "Pilates and mobility training for joint flexibility", "Adequate recovery and restorative sleep"]
    },
    "Hyperthyroidism": {
        "primary": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "muscle_weakness", "irritability", "abnormal_menstruation"],
        "secondary": ["palpitations"],
        "description": "Hyperthyroidism (overactive thyroid) is a condition occurring when the thyroid gland produces excess thyroxine hormone, accelerating body metabolism abnormally.",
        "precautions": ["Regularly monitor resting pulse rate and heart rhythm", "Avoid high-iodine foods like kelp, seaweed, and iodized contrast dyes", "Protect eyes with sunglasses and artificial tear drops if Graves' ophthalmopathy is present", "Follow endocrinologist treatment schedule strictly"],
        "medications": ["Methimazole 5mg / 10mg tab / Carbimazole 10mg tab", "Propylthiouracil (PTU) - preferred in 1st trimester pregnancy", "Propranolol 20mg / 40mg tab (beta-blocker for palpitations/tremors)", "Radioactive Iodine / Thyroidectomy (for refractory cases)"],
        "diet": ["Low-iodine diet if prepping for radioactive iodine", "Cruciferous vegetables (raw cabbage, broccoli, cauliflower contain natural goitrogens)", "Calcium and vitamin D rich foods to protect bone density", "Avoid caffeine, energy drinks, and chocolate which aggravate tachycardia"],
        "workout": ["Low-impact calming exercises: walking, restorative gentle yoga", "Avoid strenuous high-intensity interval training (HIIT) while tachycardic", "Meditation and vagus nerve breathing exercises", "Adequate caloric intake to prevent excessive muscle catabolism"]
    },
    "Hypoglycemia": {
        "primary": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "drying_and_tingling_lips", "slurred_speech", "irritability", "palpitations"],
        "secondary": ["dizziness"],
        "description": "Hypoglycemia is a condition characterized by abnormally low levels of blood glucose (typically below 70 mg/dL), commonly triggered by diabetes medication or prolonged fasting.",
        "precautions": ["Follow the 'Rule of 15': eat 15g of fast-acting sugar, wait 15 minutes, recheck blood glucose", "Always carry glucose tablets, hard candies, or fruit juice cartons", "Wear a medical alert bracelet indicating diabetes/hypoglycemia risk", "Adjust insulin or sulfonylurea dosage in consultation with doctor"],
        "medications": ["Oral Glucose Tablets (15-20g)", "Glucagon Emergency Injection kit / Nasal Powder (Baqsimi 3mg)", "Dextrose 25% or 50% IV (in hospital setting for unconscious patients)"],
        "diet": ["Immediate emergency: 1/2 cup fruit juice, 3-4 glucose candies, 1 tablespoon sugar or honey", "Follow up with complex carbohydrate + protein snack: peanut butter sandwich or cheese and crackers", "Consistent, timely meals; avoid skipping lunches or dinners", "Complex carbohydrates (oats, brown rice) for slow release"],
        "workout": ["Always check blood glucose before starting exercise (> 100 mg/dL is safe)", "Carry fast-acting carbs during all workouts", "Avoid exercising alone when blood glucose fluctuates", "Have a post-workout snack to avoid delayed nocturnal hypoglycemia"]
    },
    "Osteoarthritis": {
        "primary": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
        "secondary": ["movement_stiffness"],
        "description": "Osteoarthritis is the most common form of arthritis, caused by the progressive degradation and wear of protective cartilage at the ends of bones in weight-bearing joints.",
        "precautions": ["Maintain a healthy body weight to decrease mechanical stress on knee and hip joints", "Use assistive devices (knee braces, walking cane) when walking long distances", "Apply warm heating pads for morning stiffness and cold ice packs for acute joint swelling", "Avoid high-impact jumping and kneeling on hard surfaces"],
        "medications": ["Paracetamol 650mg / Topical Diclofenac gel", "Celecoxib 100mg / Etoricoxib 90mg tab", "Glucosamine + Chondroitin Sulfate supplement", "Intra-articular Hyaluronic Acid / Corticosteroid injections"],
        "diet": ["Omega-3 rich foods: chia seeds, walnuts, fatty fish (mackerel/salmon)", "Anti-inflammatory spices: turmeric (curcumin with black pepper) and ginger", "Bone-supportive foods: fortified almond milk, sesame seeds, leafy greens", "Avoid excessive processed sugar and fried foods which elevate inflammatory markers"],
        "workout": ["Water aerobics and swimming (zero-gravity joint movement)", "Quadriceps strengthening isometric leg raises", "Low-impact stationary cycling with seat elevated", "Gentle range-of-motion flexibility stretching"]
    },
    "Arthritis": {
        "primary": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
        "secondary": ["joint_pain"],
        "description": "Rheumatoid and inflammatory arthritis is an autoimmune condition causing painful inflammation, stiffness, and symmetric joint swelling that can lead to joint deformity.",
        "precautions": ["Early consultation with a rheumatologist for DMARD therapy initiation", "Protect small joints from mechanical strain with ergonomic tools", "Balance active periods with adequate joint rest", "Do gentle joint range-of-motion exercises every morning to loosen stiffness"],
        "medications": ["Methotrexate 7.5mg - 15mg weekly (anchor DMARD)", "Hydroxychloroquine 200mg tab", "Folic Acid 5mg tab (taken non-methotrexate days)", "Prednisolone 5mg (short bridge therapy)"],
        "diet": ["Mediterranean anti-inflammatory diet", "Antioxidant-rich berries, olive oil, and steamed green vegetables", "Omega-3 fatty acids to reduce systemic inflammation", "Limit red meat, ultra-processed foods, and refined carbohydrates"],
        "workout": ["Gentle morning mobility and finger joint range-of-motion drills", "Warm water pool hydrotherapy exercises", "Tai Chi for balance, flexibility, and mind-body calmness", "Avoid heavy resistance training during acute active flare-ups"]
    },
    "(vertigo) Paroxysmal  Positional Vertigo": {
        "primary": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
        "secondary": ["dizziness"],
        "description": "Benign Paroxysmal Positional Vertigo (BPPV) is an inner ear disorder where tiny calcium crystals (canaliths) become dislodged, causing brief, intense spinning sensation upon head turns.",
        "precautions": ["Perform the Epley maneuver or Semont maneuver under trained supervision", "Avoid sudden rapid head movements and bending backwards abruptly", "Sleep with head slightly elevated on two pillows", "Sit still immediately when spinning begins to prevent falls"],
        "medications": ["Betahistine 16mg / 24mg tab (improves microcirculation in inner ear)", "Cinnarizine 25mg tab / Dimenhydrinate 50mg", "Ondansetron 4mg tab (for nausea during acute episodes)", "Vitamin D3 supplement (deficiency links to BPPV recurrence)"],
        "diet": ["Stay well-hydrated throughout the day (at least 2.5 liters water)", "Limit high-sodium foods which disturb inner ear fluid balance", "Limit caffeine and alcohol intake which can trigger vestibulopathy", "Consistent regular nutrient-rich meals"],
        "workout": ["Brandt-Daroff vestibular rehabilitation exercises at home", "Epley repositioning maneuver performed in clinic or as instructed", "Slow gaze stabilization eye drills", "Avoid rollercoasters, gymnastics, or rapid inverted yoga poses"]
    },
    "Acne": {
        "primary": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        "secondary": ["itching"],
        "description": "Acne vulgaris is a chronic inflammatory skin condition occurring when hair follicles become plugged with dead skin cells and excess sebum oil, colonized by C. acnes bacteria.",
        "precautions": ["Wash face twice daily with a gentle salicylic acid foaming cleanser", "Do NOT pick, squeeze, or pop pimples to prevent deep scarring and hyperpigmentation", "Use non-comedogenic, oil-free moisturizers and sunscreens", "Wash pillowcases and phone screens frequently"],
        "medications": ["Benzoyl Peroxide 2.5% / 5% topical gel", "Adapalene 0.1% / Tretinoin gel (applied at night)", "Clindamycin 1% topical solution", "Oral Doxycycline 100mg (for moderate to severe inflammatory acne)"],
        "diet": ["Low glycemic index foods to avoid insulin-driven sebum surges", "Zinc and omega-3 rich pumpkin seeds and walnuts", "Limit skim cow's milk and high-sugar chocolate if they trigger outbreaks", "Drink plenty of water and herbal spearmint tea"],
        "workout": ["Shower immediately after workouts to wash away sweat and bacteria", "Wear loose, breathable athletic fabrics", "Avoid touching your face with unwashed gym hands", "Wash gym towels and headbands after every session"]
    },
    "Urinary tract infection": {
        "primary": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
        "secondary": ["stomach_pain", "mild_fever"],
        "description": "Urinary Tract Infection (UTI) is an infection in any part of your urinary system (kidneys, bladder, or urethra), most commonly caused by Escherichia coli bacteria.",
        "precautions": ["Drink at least 3 liters of water daily to flush bacteria from the urinary tract", "Wipe from front to back after using the restroom", "Urinate promptly after sexual intercourse", "Avoid harsh feminine hygiene sprays and scented soaps"],
        "medications": ["Nitrofurantoin 100mg cap (twice daily for 5-7 days)", "Fosfomycin 3g oral sachet (single dose)", "Phenazopyridine 200mg tab (urinary analgesic for burning pain)", "Disodium hydrogen citrate syrup (urinary alkalinizer)"],
        "diet": ["Pure unsweetened cranberry juice or cranberry extract capsules (contains PACs)", "Abundant water, barley water, and tender coconut water", "Probiotics like curd and kefir to support healthy urogenital microflora", "Avoid alcohol, spicy foods, caffeine, and artificial sweeteners during infection"],
        "workout": ["Mild walking and gentle movement", "Avoid swimming in public pools until infection resolves", "Avoid tight synthetic workout leggings; wear cotton underwear", "Keep workouts light during active antibiotic treatment"]
    },
    "Psoriasis": {
        "primary": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
        "secondary": ["itching"],
        "description": "Psoriasis is a chronic autoimmune skin disease that speeds up the growth cycle of skin cells, leading to thick, red, scaly plaques with silvery scales.",
        "precautions": ["Keep skin moisturized by applying thick emollient ointments right after bathing", "Take short lukewarm showers rather than hot baths", "Avoid skin injuries (cuts, sunburns) which can trigger Koebner phenomenon", "Quit smoking and moderate alcohol, which trigger severe flares"],
        "medications": ["Clobetasol Propionate 0.05% ointment", "Calcipotriol (Vitamin D analogue) topical cream", "Methotrexate 10mg - 15mg weekly (for widespread disease)", "Biologics like Secukinumab / Adalimumab (for refractory plaque psoriasis)"],
        "diet": ["Anti-inflammatory Mediterranean diet (olive oil, avocados, leafy greens)", "Omega-3 rich walnuts, flaxseeds, and fatty fish", "Eliminate alcohol, refined sugar, and processed red meats", "Turmeric and ginger in daily cooking"],
        "workout": ["Low-impact aerobic exercises: walking, cycling", "Gentle yoga and mindfulness meditation to decrease stress flares", "Moisturize skin before and after swimming or exercising", "Wear loose, friction-free soft cotton athletic wear"]
    },
    "Impetigo": {
        "primary": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
        "secondary": ["itching"],
        "description": "Impetigo is a highly contagious superficial bacterial skin infection, common in young children, characterized by honey-colored crusted sores around the mouth and nose.",
        "precautions": ["Gently wash sores with antibacterial soap and warm water, loosening crusts", "Cover sores loosely with clean gauze to prevent scratching and spreading", "Wash hands frequently and keep nails cut short", "Do not share towels, washcloths, clothing, or bedsheets with others"],
        "medications": ["Mupirocin 2% topical ointment (applied 3 times daily)", "Fusidic acid 2% cream", "Amoxicillin-Clavulanate oral syrup/tab (for widespread lesions)", "Cephalexin 500mg tab"],
        "diet": ["Zinc and Vitamin C rich citrus fruits and vegetables for skin repair", "Protein-rich nourishing broths and steamed foods", "Abundant clean water and fresh fruit smoothies", "Avoid excessively sugary treats which promote bacterial proliferation"],
        "workout": ["Isolate from contact sports, gym classes, and swimming pools until crusted over", "Gentle quiet indoor play and rest at home", "Avoid sweating which irritates crusted lesions", "Resume normal active sports only after 48 hours of antibiotic therapy"]
    }
}

# 3. Build Medicine Catalog (50+ popular drugs with categories, composition, uses, side effects, and generic substitutes)
MEDICINE_CATALOG = [
    {
        "name": "Paracetamol 650 (Dolo 650 / Calpol)",
        "salt": "Paracetamol (Acetaminophen) 650mg",
        "category": "Analgesic & Antipyretic",
        "uses": "Fever reduction, headache, toothache, muscle aches, mild-to-moderate body pain",
        "dosage": "1 tablet every 6 to 8 hours as needed (Max 3000mg/day)",
        "side_effects": "Rare at therapeutic doses; nausea, liver toxicity if taken in overdose",
        "substitutes": "Generic Paracetamol 650mg, Crocin 650, P-650, Pacimol 650",
        "price_tier": "Low (₹15 - ₹35 for 15 tabs)"
    },
    {
        "name": "Pantoprazole 40 (Pan 40 / Pantocid)",
        "salt": "Pantoprazole Sodium 40mg",
        "category": "Proton Pump Inhibitor (Antacid)",
        "uses": "GERD, acid reflux, peptic ulcers, prevention of NSAID-induced gastric irritation",
        "dosage": "1 tablet daily in the morning, 30 minutes before breakfast",
        "side_effects": "Headache, diarrhea, abdominal pain, flatulence",
        "substitutes": "Generic Pantoprazole 40mg, Pantodac 40, Protium 40, Nupenta 40",
        "price_tier": "Low-Medium (₹40 - ₹120 for 15 tabs)"
    },
    {
        "name": "Cetirizine 10 (Cetzine / Zyrtec)",
        "salt": "Cetirizine Hydrochloride 10mg",
        "category": "Antihistamine (Anti-allergic)",
        "uses": "Allergic rhinitis, sneezing, runny nose, watery eyes, hives, skin itching",
        "dosage": "1 tablet once daily, preferably at bedtime",
        "side_effects": "Mild drowsiness, dry mouth, tiredness, dizziness",
        "substitutes": "Generic Cetirizine 10mg, Alerid, Okacet, Zyncet",
        "price_tier": "Low (₹15 - ₹30 for 10 tabs)"
    },
    {
        "name": "Azithromycin 500 (Azee 500 / Azithral)",
        "salt": "Azithromycin 500mg",
        "category": "Macrolide Antibiotic",
        "uses": "Bacterial respiratory infections, tonsillitis, typhoid, sinusitis, skin infections",
        "dosage": "1 tablet once daily for 3 to 5 days, 1 hour before or 2 hours after meals",
        "side_effects": "Nausea, stomach cramps, diarrhea, temporary taste alteration",
        "substitutes": "Generic Azithromycin 500mg, Zady 500, Azax 500, ATM 500",
        "price_tier": "Medium (₹70 - ₹120 for 5 tabs)"
    },
    {
        "name": "Amoxicillin + Clavulanate 625 (Augmentin / Moxikind CV)",
        "salt": "Amoxicillin 500mg + Potassium Clavulanate 125mg",
        "category": "Broad-Spectrum Penicillin Antibiotic",
        "uses": "Bacterial bronchitis, pneumonia, sinusitis, ear infections, skin and soft tissue infections",
        "dosage": "1 tablet twice daily with food for 5 to 7 days",
        "side_effects": "Diarrhea, loose stools, nausea, mild skin rash, fungal thrush",
        "substitutes": "Generic Amox-Clav 625mg, Clavam 625, Sensiclav 625, Mega-CV 625",
        "price_tier": "Medium-High (₹120 - ₹200 for 10 tabs)"
    },
    {
        "name": "Metformin 500 (Glycomet 500 / Glucophage)",
        "salt": "Metformin Hydrochloride 500mg (Immediate or Extended Release)",
        "category": "Biguanide Antidiabetic",
        "uses": "Type 2 Diabetes Mellitus glycemic management, insulin sensitization",
        "dosage": "1 tablet once or twice daily with or immediately after meals",
        "side_effects": "Metallic taste, stomach upset, nausea, diarrhea, vitamin B12 deficiency",
        "substitutes": "Generic Metformin 500mg, Cetapin 500, Formin 500, Bigomet 500",
        "price_tier": "Very Low (₹15 - ₹40 for 20 tabs)"
    },
    {
        "name": "Telmisartan 40 (Telma 40 / Micardis)",
        "salt": "Telmisartan 40mg",
        "category": "Angiotensin II Receptor Blocker (Antihypertensive)",
        "uses": "Hypertension (high blood pressure), cardiovascular protection",
        "dosage": "1 tablet once daily in the morning or evening consistently",
        "side_effects": "Dizziness, low blood pressure upon standing, elevated blood potassium",
        "substitutes": "Generic Telmisartan 40mg, Telmikind 40, Arbitel 40, Sartel 40",
        "price_tier": "Low-Medium (₹40 - ₹90 for 15 tabs)"
    },
    {
        "name": "Amlodipine 5 (Amlong 5 / Norvasc)",
        "salt": "Amlodipine Besylate 5mg",
        "category": "Calcium Channel Blocker (Antihypertensive)",
        "uses": "Hypertension, chronic stable angina, coronary artery disease",
        "dosage": "1 tablet once daily with or without food",
        "side_effects": "Ankle swelling (peripheral edema), flushing, fatigue, dizziness",
        "substitutes": "Generic Amlodipine 5mg, Stamlo 5, Amtas 5, Primodil 5",
        "price_tier": "Very Low (₹20 - ₹45 for 15 tabs)"
    },
    {
        "name": "Levothyroxine 50 (Thyronorm 50 / Eltroxin)",
        "salt": "Levothyroxine Sodium 50mcg",
        "category": "Thyroid Hormone Replacement",
        "uses": "Hypothyroidism, goiter management, thyroid hormone deficiency",
        "dosage": "1 tablet daily on empty stomach 45 mins before breakfast with water",
        "side_effects": "Palpitations, tremor, insomnia if over-replaced",
        "substitutes": "Generic Levothyroxine 50mcg, Thyrox 50, Lethyrox 50",
        "price_tier": "Low (₹80 - ₹140 for 100 tabs)"
    },
    {
        "name": "Montelukast + Levocetirizine (Montair LC / Telekast-L)",
        "salt": "Montelukast 10mg + Levocetirizine Dihydrochloride 5mg",
        "category": "Antiasthmatic & Antiallergic",
        "uses": "Allergic rhinitis, seasonal asthma symptoms, chronic urticaria",
        "dosage": "1 tablet once daily in the evening",
        "side_effects": "Dry mouth, headache, drowsiness, vivid dreams",
        "substitutes": "Generic Montelukast-Levocetirizine, Romilast L, Levomac M, Montek LC",
        "price_tier": "Medium (₹80 - ₹160 for 10 tabs)"
    },
    {
        "name": "Fluconazole 150 (Forcan 150 / Flucos)",
        "salt": "Fluconazole 150mg",
        "category": "Antifungal (Triazole)",
        "uses": "Candidiasis, tinea cruris, ringworm fungal infections, cryptococcal infections",
        "dosage": "1 tablet once weekly or as directed by doctor",
        "side_effects": "Headache, nausea, abdominal discomfort, elevated liver enzymes",
        "substitutes": "Generic Fluconazole 150mg, Syscan 150, Zocon 150, Afung 150",
        "price_tier": "Low (₹12 - ₹25 per single tab/strip)"
    },
    {
        "name": "Omeprazole 20 (Omez / Prilosec)",
        "salt": "Omeprazole 20mg",
        "category": "Proton Pump Inhibitor",
        "uses": "Acid reflux, heartburn, duodenal ulcers, Zollinger-Ellison syndrome",
        "dosage": "1 capsule daily before meals in morning",
        "side_effects": "Mild diarrhea, flatulence, headache, abdominal pain",
        "substitutes": "Generic Omeprazole 20mg, Ocid 20, Omecip 20, Lomac 20",
        "price_tier": "Low (₹30 - ₹60 for 15 caps)"
    },
    {
        "name": "Salbutamol Inhaler (Asthalin Inhaler / Ventolin)",
        "salt": "Salbutamol (Albuterol) 100mcg per puff",
        "category": "Short-Acting Beta-2 Agonist (Bronchodilator)",
        "uses": "Acute asthma attacks, wheezing, bronchospasm, COPD relief",
        "dosage": "1 to 2 puffs inhaled when experiencing acute shortness of breath",
        "side_effects": "Fine tremors, increased heart rate, nervousness, throat irritation",
        "substitutes": "Generic Salbutamol MDI, Aerocort, Ventorlin Inhaler",
        "price_tier": "Low-Medium (₹120 - ₹180 per 200 actuations)"
    },
    {
        "name": "Ondansetron 4 (Emeset 4 / Zofran)",
        "salt": "Ondansetron Hydrochloride 4mg",
        "category": "5-HT3 Antiemetic",
        "uses": "Severe nausea and vomiting associated with gastroenteritis, chemotherapy, or surgery",
        "dosage": "1 tablet 30 minutes before meal or when nausea starts",
        "side_effects": "Constipation, headache, mild fatigue, sensation of warmth",
        "substitutes": "Generic Ondansetron 4mg, Vomikind 4, Periset 4, Onset 4",
        "price_tier": "Low (₹25 - ₹50 for 10 tabs)"
    },
    {
        "name": "Oral Rehydration Salts (ORS Sachet)",
        "salt": "Sodium Chloride, Potassium Chloride, Sodium Citrate, Dextrose (WHO formula)",
        "category": "Fluid and Electrolyte Replenisher",
        "uses": "Dehydration due to diarrhoea, vomiting, excessive sweating, heat exhaustion",
        "dosage": "Dissolve 1 full sachet in exactly 1 litre of clean drinking water; sip frequently",
        "side_effects": "Extremely safe; hypernatremia only if prepared in too little water",
        "substitutes": "Generic WHO-ORS, Electral Sachet, Walyte, Prolyte ORS",
        "price_tier": "Very Low (₹18 - ₹25 per sachet)"
    },
    {
        "name": "Ursodeoxycholic Acid 300 (Udiliv 300 / Urso)",
        "salt": "Ursodeoxycholic Acid (UDCA) 300mg",
        "category": "Hepatoprotective & Bile Acid",
        "uses": "Cholestatic liver disease, non-alcoholic fatty liver, dissolving gallstones",
        "dosage": "1 tablet twice daily with meals or glass of milk",
        "side_effects": "Soft stools, diarrhea, nausea, dizziness",
        "substitutes": "Generic UDCA 300mg, Ursocol 300, Actibile 300, Ursetor 300",
        "price_tier": "High (₹350 - ₹550 for 10 tabs)"
    },
    {
        "name": "Aceclofenac + Paracetamol (Zerodol P / Hifenac P)",
        "salt": "Aceclofenac 100mg + Paracetamol 325mg",
        "category": "NSAID Analgesic & Anti-inflammatory",
        "uses": "Joint pain, osteoarthritis, cervical spondylosis, dental pain, backache",
        "dosage": "1 tablet twice daily strictly after meals",
        "side_effects": "Gastric irritation, nausea, indigestion, heartburn",
        "substitutes": "Generic Aceclo-Para, Dolowin Plus, Acemiz, Movace P",
        "price_tier": "Low-Medium (₹45 - ₹90 for 10 tabs)"
    },
    {
        "name": "Nitrofurantoin 100 (Uribid / Martifur)",
        "salt": "Nitrofurantoin (Macrobid) 100mg",
        "category": "Urinary Antibacterial",
        "uses": "Acute uncomplicated cystitis, urinary tract infections (UTI)",
        "dosage": "1 tablet twice daily with food for 5 to 7 days",
        "side_effects": "Harmless dark brown discoloration of urine, nausea, loss of appetite",
        "substitutes": "Generic Nitrofurantoin 100mg, Niftran 100, Furadantin",
        "price_tier": "Medium (₹90 - ₹150 for 10 tabs)"
    },
    {
        "name": "Betahistine 16 (Vertin 16 / Serc)",
        "salt": "Betahistine Dihydrochloride 16mg",
        "category": "Histamine Analogue (Antivertigo)",
        "uses": "Vertigo, Meniere's disease, inner ear spinning sensation, loss of balance",
        "dosage": "1 tablet three times daily after food",
        "side_effects": "Mild headache, stomach discomfort, nausea",
        "substitutes": "Generic Betahistine 16mg, Zevert 16, Betavert 16, Vertipress 16",
        "price_tier": "Medium (₹110 - ₹180 for 15 tabs)"
    },
    {
        "name": "Atorvastatin 20 (Atorva 20 / Lipitor)",
        "salt": "Atorvastatin Calcium 20mg",
        "category": "HMG-CoA Reductase Inhibitor (Statin)",
        "uses": "Hypercholesterolemia, dyslipidemia, prevention of heart attack and stroke",
        "dosage": "1 tablet once daily at bedtime",
        "side_effects": "Mild muscle aches, elevated liver enzymes, digestive distress",
        "substitutes": "Generic Atorvastatin 20mg, Storvas 20, Tonact 20, Lipikind 20",
        "price_tier": "Low-Medium (₹50 - ₹110 for 15 tabs)"
    }
]

# 4. Drug-Drug Interactions Database (Safety checker)
DRUG_INTERACTIONS = [
    {
        "drug1": "Aspirin",
        "drug2": "Ibuprofen / Aceclofenac",
        "severity": "High",
        "description": "Concurrent use of multiple NSAIDs exponentially increases the risk of severe gastrointestinal ulceration, stomach bleeding, and renal impairment.",
        "action": "Avoid combining. If pain control is needed, substitute with Paracetamol or use gastroprotective PPI."
    },
    {
        "drug1": "Metformin",
        "drug2": "Alcohol",
        "severity": "High",
        "description": "Alcohol potentiates Metformin's effect on lactate metabolism, drastically elevating the life-threatening risk of Lactic Acidosis and severe hypoglycemia.",
        "action": "Strictly avoid heavy or binge alcohol intake while on Metformin therapy."
    },
    {
        "drug1": "Telmisartan",
        "drug2": "Spironolactone / Potassium Supplements",
        "severity": "High",
        "description": "Both Telmisartan and Spironolactone conserve serum potassium. Combining them can lead to dangerous Hyperkalemia (fatal cardiac arrhythmias).",
        "action": "Regularly monitor serum potassium and creatinine. Dosage adjustment needed."
    },
    {
        "drug1": "Paracetamol",
        "drug2": "Alcohol",
        "severity": "Moderate",
        "description": "Chronic alcohol consumption induces CYP2E1 liver enzymes, producing excess hepatotoxic NAPQI metabolite from Paracetamol, risking liver injury.",
        "action": "Limit Paracetamol to maximum 2000mg/day and avoid alcohol while taking fever/pain meds."
    },
    {
        "drug1": "Pantoprazole / Omeprazole",
        "drug2": "Levothyroxine",
        "severity": "Moderate",
        "description": "Proton pump inhibitors decrease stomach acidity required for optimal oral absorption of Levothyroxine, potentially causing uncontrolled hypothyroidism.",
        "action": "Take Levothyroxine on empty stomach at least 4 hours apart from PPIs."
    },
    {
        "drug1": "Azithromycin",
        "drug2": "Ondansetron",
        "severity": "Moderate",
        "description": "Both drugs can prolong the cardiac QTc interval on ECG, increasing the risk of Torsades de Pointes ventricular arrhythmias.",
        "action": "Use with caution in cardiac patients; monitor ECG if long-term or high doses are used."
    },
    {
        "drug1": "Amlodipine",
        "drug2": "Simvastatin / Atorvastatin",
        "severity": "Moderate",
        "description": "Amlodipine inhibits CYP3A4 metabolism of statins, elevating statin blood levels and increasing the risk of rhabdomyolysis (muscle breakdown).",
        "action": "Cap Simvastatin dose at 20mg daily or switch to Rosuvastatin / monitored Atorvastatin."
    },
    {
        "drug1": "Cetirizine / Levocetirizine",
        "drug2": "Alcohol / Sedatives",
        "severity": "Moderate",
        "description": "Additive central nervous system depression leading to profound drowsiness, impaired motor reflexes, and respiratory depression.",
        "action": "Do not drive, operate machinery, or consume alcohol while on sedating antihistamines."
    },
    {
        "drug1": "Methotrexate",
        "drug2": "Aceclofenac / Ibuprofen",
        "severity": "High",
        "description": "NSAIDs decrease renal clearance of Methotrexate, causing toxic blood levels of Methotrexate with severe bone marrow suppression.",
        "action": "Strictly contraindicated without close specialist rheumatology monitoring."
    },
    {
        "drug1": "Ciprofloxacin / Levofloxacin",
        "drug2": "Antacids (Aluminium / Magnesium)",
        "severity": "Moderate",
        "description": "Multivalent cations in antacids bind with fluoroquinolone antibiotics in the gut, reducing antibiotic absorption by up to 80%.",
        "action": "Administer antibiotics at least 2 hours before or 4 hours after taking antacids."
    }
]

# Generate synthetic realistic training & testing CSV data
def generate_symptom_disease_datasets():
    training_rows = []
    testing_rows = []

    for disease, meta in DISEASE_DATA.items():
        primary = meta["primary"]
        secondary = meta.get("secondary", [])

        # Generate 120 training variations per disease
        for _ in range(120):
            row = {sym: 0 for sym in SYMPTOMS}
            # Always include 70-100% of primary symptoms
            n_prim = max(2, int(len(primary) * random.uniform(0.7, 1.0)))
            selected_prim = random.sample(primary, min(n_prim, len(primary)))
            for s in selected_prim:
                if s in row:
                    row[s] = 1

            # Sometimes include 1-2 secondary symptoms
            if secondary and random.random() > 0.35:
                n_sec = random.randint(1, min(2, len(secondary)))
                for s in random.sample(secondary, n_sec):
                    if s in row:
                        row[s] = 1

            # Very rarely include 1 noise symptom (real life patient complaint noise)
            if random.random() < 0.05:
                random_noise = random.choice(SYMPTOMS)
                row[random_noise] = 1

            row["prognosis"] = disease
            training_rows.append(row)

        # Generate 10 testing variations per disease
        for _ in range(10):
            row = {sym: 0 for sym in SYMPTOMS}
            n_prim = max(2, int(len(primary) * random.uniform(0.8, 1.0)))
            for s in random.sample(primary, min(n_prim, len(primary))):
                if s in row:
                    row[s] = 1
            if secondary and random.random() > 0.4:
                for s in random.sample(secondary, min(1, len(secondary))):
                    if s in row:
                        row[s] = 1
            row["prognosis"] = disease
            testing_rows.append(row)

    df_train = pd.DataFrame(training_rows)
    df_test = pd.DataFrame(testing_rows)

    # Shuffle datasets
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    df_train.to_csv(os.path.join(DATA_DIR, "Training.csv"), index=False)
    df_test.to_csv(os.path.join(DATA_DIR, "Testing.csv"), index=False)
    print(f"Generated Training.csv: {df_train.shape} rows x cols")
    print(f"Generated Testing.csv: {df_test.shape} rows x cols")

# Generate Auxiliary CSVs
def generate_auxiliary_csvs():
    # 1. Descriptions
    desc_rows = [{"Disease": dis, "Description": meta["description"]} for dis, meta in DISEASE_DATA.items()]
    pd.DataFrame(desc_rows).to_csv(os.path.join(DATA_DIR, "descriptions.csv"), index=False)

    # 2. Precautions
    prec_rows = []
    for dis, meta in DISEASE_DATA.items():
        precs = meta["precautions"]
        row = {"Disease": dis}
        for i, p in enumerate(precs):
            row[f"Precaution_{i+1}"] = p
        prec_rows.append(row)
    pd.DataFrame(prec_rows).to_csv(os.path.join(DATA_DIR, "precautions.csv"), index=False)

    # 3. Medications
    med_rows = []
    for dis, meta in DISEASE_DATA.items():
        med_rows.append({
            "Disease": dis,
            "Medication": str(meta["medications"])
        })
    pd.DataFrame(med_rows).to_csv(os.path.join(DATA_DIR, "medications.csv"), index=False)

    # 4. Diets
    diet_rows = []
    for dis, meta in DISEASE_DATA.items():
        diet_rows.append({
            "Disease": dis,
            "Diet": str(meta["diet"])
        })
    pd.DataFrame(diet_rows).to_csv(os.path.join(DATA_DIR, "diets.csv"), index=False)

    # 5. Workouts
    workout_rows = []
    for dis, meta in DISEASE_DATA.items():
        workout_rows.append({
            "Disease": dis,
            "Workout": str(meta["workout"])
        })
    pd.DataFrame(workout_rows).to_csv(os.path.join(DATA_DIR, "workouts.csv"), index=False)

    # 6. Medicine Catalog
    pd.DataFrame(MEDICINE_CATALOG).to_csv(os.path.join(DATA_DIR, "medicine_catalog.csv"), index=False)

    # 7. Drug Interactions
    pd.DataFrame(DRUG_INTERACTIONS).to_csv(os.path.join(DATA_DIR, "drug_interactions.csv"), index=False)

    print("Generated all auxiliary medical databases successfully!")

if __name__ == "__main__":
    generate_symptom_disease_datasets()
    generate_auxiliary_csvs()
