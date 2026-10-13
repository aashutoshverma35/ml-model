"""
ml_model.py
===========
Multi-Disease Symptom Intelligence Module for MediCare ML.
Handles model loading, symptom categorization, multi-hot feature vectorization,
multi-class disease inference, differential diagnosis calculation, and clinical advice.
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "symptom_disease_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "symptom_features.json")
REPORT_PATH = os.path.join(MODELS_DIR, "symptom_evaluation_report.json")

# In-memory cached model and feature list
_CACHED_MODEL = None
_CACHED_FEATURES = None


# -----------------------------------------------------------------------------
# Disease Clinical Knowledge Base (41 Distinct Conditions)
# -----------------------------------------------------------------------------
DISEASE_METADATA = {
    "Fungal infection": {
        "department": "Dermatology",
        "severity": "Mild",
        "description": "A superficial fungal infection of the skin or nails caused by dermatophytes or yeast.",
        "precautions": ["Keep infected area clean and dry", "Use prescribed antifungal topical creams", "Avoid sharing towels and personal clothing", "Wear loose cotton garments"]
    },
    "Allergy": {
        "department": "General Medicine",
        "severity": "Mild",
        "description": "An immune hypersensitivity reaction triggered by allergens such as pollen, dust, or food items.",
        "precautions": ["Identify and avoid triggering allergens", "Consider non-drowsy antihistamines", "Use air purifiers indoors", "Seek emergency care if airway swells"]
    },
    "GERD": {
        "department": "Gastroenterology",
        "severity": "Moderate",
        "description": "Gastroesophageal Reflux Disease where stomach acid persistently flows backward into the esophagus.",
        "precautions": ["Avoid spicy, fatty, and acidic foods", "Do not lie down for 2-3 hours after eating", "Elevate head of bed while sleeping", "Eat smaller, more frequent meals"]
    },
    "Chronic cholestasis": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "Impaired or reduced bile flow from the liver to the duodenum, leading to bile accumulation.",
        "precautions": ["Consult a gastroenterologist or hepatologist", "Follow low-fat dietary recommendations", "Monitor liver enzyme panel regularly", "Avoid alcohol consumption entirely"]
    },
    "Drug Reaction": {
        "department": "Dermatology",
        "severity": "Moderate",
        "description": "An adverse immune or pharmacological cutaneous reaction caused by ingested medications.",
        "precautions": ["Discontinue suspected medication under doctor supervision", "Take cool baths to soothe skin irritation", "Stay hydrated and avoid sun exposure", "Consult an allergist immediately"]
    },
    "Peptic ulcer diseae": {
        "department": "Gastroenterology",
        "severity": "Moderate",
        "description": "Painful sores or erosions on the inner lining of the stomach or small intestine.",
        "precautions": ["Avoid NSAID pain relievers (like ibuprofen)", "Limit caffeine and alcohol intake", "Take prescribed proton-pump inhibitors", "Consume non-irritating, bland meals"]
    },
    "AIDS": {
        "department": "Infectious Diseases",
        "severity": "High Attention",
        "description": "Advanced HIV infection causing progressive deterioration of the immune system.",
        "precautions": ["Adhere strictly to Antiretroviral Therapy (ART)", "Practice safe barrier protection", "Maintain sterile hygiene and vaccinations", "Schedule regular CD4 and viral load counts"]
    },
    "Diabetes": {
        "department": "Endocrinology",
        "severity": "Moderate",
        "description": "A chronic metabolic condition characterized by elevated blood glucose and insulin resistance.",
        "precautions": ["Monitor fasting and post-meal glucose levels", "Engage in regular aerobic physical exercise", "Follow a low-glycemic, fiber-rich diet", "Maintain optimal foot care and hydration"]
    },
    "Gastroenteritis": {
        "department": "General Medicine",
        "severity": "Moderate",
        "description": "Inflammation of the gastrointestinal tract commonly known as stomach flu or food poisoning.",
        "precautions": ["Rehydrate thoroughly with oral rehydration solutions (ORS)", "Follow the BRAT diet (bananas, rice, applesauce, toast)", "Avoid dairy, greasy foods, and caffeine", "Rest adequately"]
    },
    "Bronchial Asthma": {
        "department": "Pulmonology",
        "severity": "High Attention",
        "description": "Chronic inflammatory airway disease causing episodic wheezing, chest tightness, and dyspnea.",
        "precautions": ["Carry rapid-relief inhaler at all times", "Avoid cold air, dust, smoke, and pollen", "Use peak flow meter to monitor lung volume", "Take controller medications as prescribed"]
    },
    "Hypertension": {
        "department": "Cardiology",
        "severity": "Moderate",
        "description": "Persistently elevated arterial blood pressure putting strain on cardiovascular vessels.",
        "precautions": ["Reduce dietary sodium intake (<2g/day)", "Engage in 30 minutes of daily brisk walking", "Manage emotional stress with meditation", "Monitor resting blood pressure daily"]
    },
    "Migraine": {
        "department": "Neurology",
        "severity": "Moderate",
        "description": "A neurological condition causing intense, throbbing unilateral headache often accompanied by nausea and photophobia.",
        "precautions": ["Rest in a quiet, dark, well-ventilated room", "Apply a cold compress to forehead or temples", "Identify food triggers (aged cheeses, chocolate)", "Maintain consistent sleep and meal schedules"]
    },
    "Cervical spondylosis": {
        "department": "Orthopedics",
        "severity": "Mild",
        "description": "Age-related wear and tear of the spinal discs and joints in the cervical neck region.",
        "precautions": ["Practice gentle neck isometric exercises", "Use ergonomic supportive pillows and posture aids", "Avoid prolonged downward head tilting at screens", "Apply intermittent warm compress"]
    },
    "Paralysis (brain hemorrhage)": {
        "department": "Neurology",
        "severity": "High Attention",
        "description": "Acute cerebrovascular event (stroke) causing localized neurological deficit and muscle paralysis.",
        "precautions": ["Seek emergency neuro-critical care immediately (Call emergency services)", "Do not administer oral food/fluids to prevent aspiration", "Monitor airway and vital signs", "Begin early neuro-rehabilitation"]
    },
    "Jaundice": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "Yellowish pigmentation of skin and sclera caused by high levels of bilirubin from hepatic dysfunction.",
        "precautions": ["Consult a physician for liver function tests (LFT)", "Drink plenty of boiled water and clear fluids", "Avoid oily, fried, and heavy foods", "Rest completely and avoid alcohol"]
    },
    "Malaria": {
        "department": "General Medicine",
        "severity": "High Attention",
        "description": "Mosquito-borne parasitic infection characterized by recurring cyclical high fevers, shaking chills, and profuse sweating.",
        "precautions": ["Get a rapid diagnostic blood smear test immediately", "Complete full course of prescribed antimalarials", "Use mosquito repellents and bed nets", "Stay well hydrated and take antipyretics"]
    },
    "Chicken pox": {
        "department": "General Medicine",
        "severity": "Moderate",
        "description": "Highly contagious viral infection caused by varicella-zoster presenting with an itchy vesicular rash.",
        "precautions": ["Isolate until all blisters have crusted over", "Apply calamine lotion to soothe itching", "Take lukewarm oatmeal baths", "Avoid scratching to prevent secondary bacterial infection"]
    },
    "Dengue": {
        "department": "General Medicine",
        "severity": "High Attention",
        "description": "Mosquito-transmitted viral infection marked by high fever, severe retro-orbital headache, and profound joint/muscle pain ('breakbone fever').",
        "precautions": ["Monitor platelet counts daily via complete blood count (CBC)", "Maintain intensive oral hydration (water, ORS, coconut water)", "Avoid aspirin or ibuprofen (risk of hemorrhage)", "Seek hospital admission if persistent vomiting occurs"]
    },
    "Typhoid": {
        "department": "General Medicine",
        "severity": "High Attention",
        "description": "Bacterial infection caused by Salmonella typhi, spreading through contaminated water or food.",
        "precautions": ["Drink only boiled or purified water", "Consume well-cooked, hot, hygienic meals", "Complete prescribed antibiotic course fully", "Wash hands thoroughly with soap before meals"]
    },
    "hepatitis A": {
        "department": "Gastroenterology",
        "severity": "Moderate",
        "description": "Highly contagious acute liver infection transmitted primarily through the fecal-oral route.",
        "precautions": ["Rest and avoid strenuous physical activity", "Drink plenty of clean fluids and fruit juices", "Strictly avoid alcohol and hepatotoxic drugs", "Practice meticulous hand hygiene"]
    },
    "Hepatitis B": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "Viral liver infection transmitted through blood and body fluids that can lead to chronic liver disease.",
        "precautions": ["Consult a hepatologist for viral load evaluation", "Ensure close contacts receive Hepatitis B vaccination", "Avoid sharing razors, toothbrushes, or needles", "Avoid alcohol completely"]
    },
    "Hepatitis C": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "Blood-borne viral infection causing chronic hepatic inflammation, now treatable with direct-acting antivirals.",
        "precautions": ["Consult a specialist for modern direct antiviral therapy", "Avoid alcohol and hepatotoxic substances", "Get screened for liver fibrosis", "Maintain a balanced liver-friendly diet"]
    },
    "Hepatitis D": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "A serious liver infection caused by the hepatitis D virus that only occurs alongside Hepatitis B.",
        "precautions": ["Specialist hepatology monitoring is essential", "Strict compliance with antiviral regimen", "Avoid alcohol and unverified supplements", "Routine ultrasound surveillance"]
    },
    "Hepatitis E": {
        "department": "Gastroenterology",
        "severity": "Moderate",
        "description": "Waterborne viral hepatitis commonly causing self-limiting acute liver inflammation.",
        "precautions": ["Consume only boiled or filtered drinking water", "Ensure thorough cooking of meats and seafood", "Pregnant individuals require close clinical monitoring", "Avoid fatty foods"]
    },
    "Alcoholic hepatitis": {
        "department": "Gastroenterology",
        "severity": "High Attention",
        "description": "Acute liver inflammation caused by excessive and prolonged alcohol consumption.",
        "precautions": ["Complete, permanent cessation of all alcoholic beverages", "Seek structured addiction and medical support", "Follow a high-protein, nutritionally dense diet", "Regular screening for hepatic cirrhosis"]
    },
    "Tuberculosis": {
        "department": "Pulmonology",
        "severity": "High Attention",
        "description": "Infectious bacterial disease primarily affecting lungs, marked by persistent cough, hemoptysis, and night sweats.",
        "precautions": ["Strictly adhere to the complete 6-month DOTS antibiotic protocol", "Wear a protective mask in public spaces", "Ensure room has proper sunlight and ventilation", "Eat a high-protein, calorie-rich diet"]
    },
    "Common Cold": {
        "department": "General Medicine",
        "severity": "Mild",
        "description": "Mild viral upper respiratory tract infection characterized by sneezing, nasal congestion, and sore throat.",
        "precautions": ["Get plenty of rest and sleep", "Drink warm fluids like herbal tea, warm water, and soups", "Use saline nasal sprays or steam inhalation", "Wash hands regularly to prevent spread"]
    },
    "Pneumonia": {
        "department": "Pulmonology",
        "severity": "High Attention",
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid or purulent material.",
        "precautions": ["Consult a pulmonologist for chest X-ray and antibiotic/antiviral treatment", "Monitor oxygen saturation with a pulse oximeter", "Perform deep breathing exercises", "Rest in an elevated upright position"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "department": "General Surgery",
        "severity": "Mild",
        "description": "Swollen veins in the lower rectum and anus causing discomfort, itching, and bleeding.",
        "precautions": ["Increase dietary fiber intake (vegetables, oats, psyllium husk)", "Drink at least 2.5 - 3 liters of water daily", "Take warm sitz baths for 15 minutes twice daily", "Avoid straining during bowel movements"]
    },
    "Heart attack": {
        "department": "Cardiology",
        "severity": "High Attention",
        "description": "Medical emergency resulting from occlusion of coronary blood flow to a section of the heart muscle.",
        "precautions": ["Call emergency medical services immediately (Cardiac Emergency)", "Chew aspirin if instructed by medical dispatcher", "Keep patient calm and seated", "Do not attempt home remedies"]
    },
    "Varicose veins": {
        "department": "Vascular Surgery",
        "severity": "Mild",
        "description": "Enlarged, twisted, visible veins most commonly occurring in the lower extremities due to valve insufficiency.",
        "precautions": ["Wear graduated compression stockings", "Elevate legs above heart level when resting", "Avoid prolonged standing or sitting without movement", "Engage in low-impact walking exercises"]
    },
    "Hypothyroidism": {
        "department": "Endocrinology",
        "severity": "Moderate",
        "description": "Underactive thyroid gland producing insufficient thyroid hormones, leading to fatigue and weight gain.",
        "precautions": ["Take prescribed levothyroxine daily on an empty stomach", "Check serum TSH and Free T4 levels every 3-6 months", "Maintain balanced intake of dietary selenium and zinc", "Exercise regularly"]
    },
    "Hyperthyroidism": {
        "department": "Endocrinology",
        "severity": "Moderate",
        "description": "Overactive thyroid gland producing excessive thyroid hormones, causing palpitations, tremors, and weight loss.",
        "precautions": ["Take prescribed antithyroid medications consistently", "Avoid excessive iodine foods (seaweed, iodized salts in excess)", "Practice relaxation techniques to reduce heart rate", "Monitor resting pulse"]
    },
    "Hypoglycemia": {
        "department": "Endocrinology",
        "severity": "High Attention",
        "description": "Dangerously low blood glucose concentration (<70 mg/dL) causing shakiness, sweating, and confusion.",
        "precautions": ["Apply the 15-15 rule: consume 15 grams of fast-acting sugar (fruit juice, candy)", "Recheck blood glucose after 15 minutes", "Eat a balanced meal once glucose stabilizes", "Carry emergency glucose tablets"]
    },
    "Osteoarthristis": {
        "department": "Orthopedics",
        "severity": "Moderate",
        "description": "Degenerative joint disease caused by gradual breakdown of cartilage in knees, hips, or hands.",
        "precautions": ["Perform low-impact exercises like swimming and cycling", "Maintain a healthy body weight to reduce joint load", "Apply heat packs before activity and ice after", "Use supportive footwear"]
    },
    "Arthritis": {
        "department": "Rheumatology",
        "severity": "Moderate",
        "description": "Inflammation of one or more joints causing stiffness, swelling, and reduced range of motion.",
        "precautions": ["Consult a rheumatologist for anti-inflammatory treatment", "Engage in gentle daily range-of-motion stretching", "Eat an anti-inflammatory diet rich in Omega-3s", "Apply warm compresses to stiff joints"]
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "department": "ENT",
        "severity": "Moderate",
        "description": "Benign inner ear disorder causing brief episodes of mild to intense spinning sensations when changing head position.",
        "precautions": ["Perform Epley canalith repositioning maneuver with an ENT", "Avoid sudden head movements or turning quickly in bed", "Sit down immediately when dizzy to prevent falls", "Sleep with head slightly elevated"]
    },
    "Acne": {
        "department": "Dermatology",
        "severity": "Mild",
        "description": "Inflammatory skin condition occurring when hair follicles become plugged with sebum and dead skin cells.",
        "precautions": ["Wash face twice daily with mild salicylic acid cleanser", "Avoid touching or squeezing blemishes", "Use non-comedogenic, oil-free moisturizers and sunscreens", "Keep pillowcases clean"]
    },
    "Urinary tract infection": {
        "department": "Urology",
        "severity": "Moderate",
        "description": "Bacterial infection in any part of the urinary system (kidneys, ureters, bladder, or urethra).",
        "precautions": ["Drink plenty of water to flush bacteria out", "Complete prescribed antibiotic course without skipping", "Avoid caffeine, alcohol, and spicy foods that irritate bladder", "Urinate promptly when feeling the urge"]
    },
    "Psoriasis": {
        "department": "Dermatology",
        "severity": "Moderate",
        "description": "Autoimmune skin disease causing rapid cell turnover resulting in thick, silvery, scaly plaques.",
        "precautions": ["Moisturize skin heavily with thick emollient ointments", "Get moderate, controlled sun exposure", "Avoid known triggers like emotional stress and cold weather", "Use prescribed topical steroids or biologics"]
    },
    "Impetigo": {
        "department": "Dermatology",
        "severity": "Mild",
        "description": "Contagious superficial bacterial skin infection causing honey-colored crusted sores around mouth and nose.",
        "precautions": ["Gently clean sores with mild soap and warm water", "Apply prescribed mupirocin antibiotic ointment", "Avoid touching or picking at crusted lesions", "Wash hands frequently and use separate towels"]
    }
}


# -----------------------------------------------------------------------------
# Categorized Symptoms Dictionary for Intuitive Web UI
# -----------------------------------------------------------------------------
SYMPTOM_CATEGORIES = {
    "General & Systemic": [
        "high_fever", "mild_fever", "chills", "shivering", "sweating", "fatigue",
        "malaise", "lethargy", "weight_loss", "restlessness", "toxic_look_(typhos)"
    ],
    "Respiratory": [
        "cough", "breathlessness", "continuous_sneezing", "chest_pain", "phlegm",
        "throat_irritation", "sinus_pressure", "runny_nose", "congestion",
        "blood_in_sputum", "mucoid_sputum", "rusty_sputum"
    ],
    "Digestive & Gastrointestinal": [
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "indigestion",
        "nausea", "loss_of_appetite", "constipation", "abdominal_pain", "diarrhoea",
        "passage_of_gases", "belly_pain", "distention_of_abdomen", "stomach_bleeding"
    ],
    "Skin, Face & External": [
        "itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin",
        "bruising", "swollen_legs", "puffy_face_and_eyes", "enlarged_thyroid",
        "brittle_nails", "swollen_extremeties", "drying_and_tingling_lips",
        "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
        "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
        "blister", "red_sore_around_nose", "yellow_crust_ooze"
    ],
    "Musculoskeletal & Pain": [
        "joint_pain", "muscle_wasting", "back_pain", "muscle_weakness",
        "neck_pain", "knee_pain", "hip_joint_pain", "muscle_pain",
        "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"
    ],
    "Neurological & Sensory": [
        "headache", "dizziness", "cramps", "loss_of_balance", "unsteadiness",
        "weakness_of_one_body_side", "altered_sensorium", "lack_of_concentration",
        "visual_disturbances", "loss_of_smell", "depression", "irritability"
    ],
    "Urinary & Metabolic": [
        "burning_micturition", "spotting_urination", "dark_urine", "yellow_urine",
        "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
        "swelling_of_stomach", "swollen_blood_vessels", "excessive_hunger",
        "extra_marital_contacts", "polyuria", "family_history", "cold_hands_and_feets",
        "mood_swings", "weight_gain", "fast_heart_rate", "pain_during_bowel_movements",
        "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain",
        "weakness_in_limbs", "palpitations", "internal_itching", "red_spots_over_body",
        "abnormal_menstruation", "dischromic__patches", "watering_from_eyes",
        "increased_appetite", "foul_smell_of_urine", "continuous_feel_of_urine",
        "skin_rash", "redness_of_eyes"
    ]
}


# -----------------------------------------------------------------------------
# Quick-Demo Viva Presets (Instantly populates common diseases)
# -----------------------------------------------------------------------------
DEMO_PRESETS = {
    "malaria": {
        "title": "Malaria / Parasitic Infection",
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        "expected": "Malaria"
    },
    "dengue": {
        "title": "Dengue Fever",
        "symptoms": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "loss_of_appetite"],
        "expected": "Dengue"
    },
    "cold": {
        "title": "Common Cold / Upper Respiratory",
        "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose", "congestion"],
        "expected": "Common Cold"
    },
    "gerd": {
        "title": "GERD / Acid Reflux",
        "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
        "expected": "GERD"
    },
    "pneumonia": {
        "title": "Pneumonia / Lung Infection",
        "symptoms": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain"],
        "expected": "Pneumonia"
    },
    "fungal": {
        "title": "Fungal Skin Infection",
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic__patches"],
        "expected": "Fungal infection"
    }
}


def load_model():
    """Load the trained Random Forest model and feature list."""
    global _CACHED_MODEL, _CACHED_FEATURES
    if _CACHED_MODEL is not None and _CACHED_FEATURES is not None:
        return _CACHED_MODEL, _CACHED_FEATURES

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise RuntimeError("Model or features not found. Run 'python train_model.py' first.")

    _CACHED_MODEL = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        _CACHED_FEATURES = json.load(f)

    return _CACHED_MODEL, _CACHED_FEATURES


def load_evaluation_metrics():
    """Load evaluation report JSON if available."""
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def format_symptom_name(symptom_key):
    """Convert snake_case symptom key into human-readable Title Case."""
    return symptom_key.replace("__", " ").replace("_", " ").title()


def get_all_symptoms():
    """Return list of all 132 available symptoms with formatted labels."""
    _, features = load_model()
    return [{"key": feat, "label": format_symptom_name(feat)} for feat in features]


def predict_disease_from_symptoms(selected_symptoms_list):
    """
    Given a list of symptom keys, format into a 132-dimension multi-hot vector,
    run inference through the Random Forest classifier, and return the predicted
    disease, probability scores, differential diagnoses, and medical advice.
    """
    model, features = load_model()

    if not selected_symptoms_list or len(selected_symptoms_list) < 2:
        raise ValueError("Please select at least 2 distinct symptoms for reliable disease prediction.")

    # Construct multi-hot vector of length 132
    vector = [1 if feat in selected_symptoms_list else 0 for feat in features]
    X_input = pd.DataFrame([vector], columns=features)

    # Inference: predict class and class probabilities
    pred_class = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Sort classes by descending probability
    sorted_indices = np.argsort(probabilities)[::-1]

    top_disease = str(pred_class)
    top_prob = float(probabilities[sorted_indices[0]])
    top_pct = round(top_prob * 100, 1)

    # Differential diagnoses (top 3)
    differentials = []
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        prob_val = float(probabilities[idx])
        if prob_val > 0.01:
            differentials.append({
                "disease": str(classes[idx]),
                "probability": round(prob_val * 100, 1)
            })

    # Retrieve disease metadata
    meta = DISEASE_METADATA.get(top_disease, {
        "department": "General Medicine",
        "severity": "Moderate",
        "description": "A clinical condition matching the observed symptom cluster in the machine learning knowledgebase.",
        "precautions": ["Consult a medical physician for formal evaluation", "Stay well-hydrated and rest", "Avoid unverified self-medication"]
    })

    # Formatted symptoms list
    formatted_symptoms = [format_symptom_name(s) for s in selected_symptoms_list]

    return {
        "predicted_disease": top_disease,
        "confidence_pct": top_pct,
        "severity": meta["severity"],
        "department": meta["department"],
        "description": meta["description"],
        "precautions": meta["precautions"],
        "differentials": differentials,
        "selected_symptoms": formatted_symptoms,
        "raw_symptoms": selected_symptoms_list,
        "symptom_count": len(selected_symptoms_list),
        "model_name": "Multi-Class Random Forest Disease Classifier (132 Symptoms, 41 Diseases)",
        "disclaimer": (
            "Educational Demonstration Only: MediCare ML's symptom-based disease prediction "
            "is generated by a machine learning model for university engineering coursework. "
            "It does NOT constitute a licensed medical diagnosis. Always consult certified medical doctors."
        )
    }
