# MediCare ML — Multi-Disease Symptom Diagnostic & Management System

**MediCare ML** is a full-stack educational healthcare web application built with **Python, Flask, SQLite, Bootstrap 5, and Scikit-Learn**. It allows patients and healthcare coordinators to manage medical records, book appointments, and run an automated **Machine Learning Symptom Diagnostic Engine** that evaluates observable symptoms across 132 indicators to predict among **41 distinct human conditions** (e.g. *Malaria, Dengue, Common Cold, GERD, Pneumonia, Typhoid, Fungal Infection, etc.*).

---

## 1. Key Features

- **Executive Clinical Dashboard (`/`)**: Real-time statistics queried directly from SQLite (**Total Patients**, **Clinical Appointments**, **Total Screenings**, and **High Attention Cases**) with quick actions and recent diagnostic logs.
- **Patient Management System (`/patients`)**: Full CRUD operations (Create, Read, Update, Delete) for patient demographic and medical records with instant search filtering.
- **Multi-Disease Symptom Diagnostic Engine (`/predict`)**:
  - Interactive **Symptom Picker** organized across 7 anatomical/clinical categories (*General, Respiratory, Digestive, Skin, Musculoskeletal, Neurological, Urinary/Metabolic*).
  - Instant **Search Filter** across all 132 symptoms.
  - **1-Click Quick-Demo Presets** (*Malaria*, *Dengue*, *Common Cold*, *GERD*, *Pneumonia*, *Fungal Infection*) for fast college viva demonstrations.
  - Evaluates symptom clusters using a supervised **Multi-Class Random Forest Classifier** trained on the 4,920-record Disease-Symptom dataset.
- **Comprehensive Diagnostic Output (`/result/<id>`)**:
  - Primary predicted disease with clinical severity badge (*Mild*, *Moderate*, *High Attention*).
  - Model confidence match percentage.
  - **Top Differential Diagnoses** ranking the 2nd and 3rd most probable conditions.
  - Recommended clinical precautions and home-care measures.
  - **Specialist Department Referral** with a 1-click **"Book Appointment with this Specialist"** button that auto-populates the appointment desk!
  - Print / Save report option (`window.print()`).
- **Appointment Scheduling Desk (`/appointments`)**: Schedule specialist doctor consultations, filter by status (`Scheduled`, `Completed`, `Cancelled`), and manage appointments.
- **Diagnostic History Log (`/history`)**: Filterable log of all past multi-disease screenings with disease-type and patient search filters.
- **Academic Documentation & Viva Guide (`/about`)**: Architecture diagrams, model evaluation metrics, and oral exam viva cheat-sheet.

---

## 2. Technology Stack

- **Backend Framework**: Python 3 & Flask
- **Frontend & Styling**: HTML5, CSS3, Bootstrap 5 (CDN), Bootstrap Icons (CDN), custom CSS (`static/css/style.css`)
- **Database**: SQLite (serverless database in `instance/medicare.db` with parameterized queries)
- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Testing**: Python standard library `unittest`

---

## 3. Project File Structure

```text
medicare_ml/
│
├── app.py                  # Main Flask application controller & routes
├── ml_model.py             # Multi-disease symptom inference & knowledgebase
├── database.py             # SQLite helper, schema creation & demo seeding
├── train_model.py          # Random Forest training & evaluation script
├── test_app.py             # Automated end-to-end unit & integration test suite
├── requirements.txt        # Python dependency manifest
├── README.md               # Beginner-friendly setup & viva documentation
│
├── data/
│   └── symptoms_disease.csv # 132 symptoms, 41 diseases dataset (4,920 records)
│
├── models/
│   ├── symptom_disease_model.joblib # Trained Random Forest model artifact
│   ├── symptom_features.json        # 132 symptom names in feature order
│   └── symptom_evaluation_report.json # Genuine computed evaluation metrics report
│
├── static/
│   ├── css/
│   │   └── style.css       # Clean healthcare color palette & responsive styles
│   └── js/
│       └── main.js         # Mobile drawer toggle & UI helpers
│
├── templates/
│   ├── base.html           # Master layout with sidebar & top navigation
│   ├── dashboard.html      # KPI cards & recent multi-disease activity
│   ├── patients.html       # Patient directory with search & CRUD actions
│   ├── patient_form.html   # Patient creation & editing form
│   ├── patient_detail.html # Individual patient health profile & timeline
│   ├── predict.html        # Categorized symptom picker with quick presets
│   ├── result.html         # Diagnostic output, differentials & specialist referral
│   ├── appointments.html   # Appointments list with status filters
│   ├── appointment_form.html # Appointment booking form
│   ├── history.html        # Multi-disease historical logs with search
│   ├── about.html          # ML architecture, 41 diseases list & viva guide
│   └── error.html          # Custom user-friendly 404 & 500 error views
│
└── instance/
    └── medicare.db         # Auto-generated SQLite database
```

---

## 4. How to Run the Application (Windows)

```powershell
# 1. Open PowerShell and navigate to the project
cd C:\Users\hp\.gemini\antigravity\scratch\medicare_ml

# 2. Run the automated test suite (verifies 9 test cases)
python test_app.py

# 3. Start the Flask application
python app.py
```

Open your browser and visit:
**`http://127.0.0.1:5000`**

---

## 5. Machine Learning Evaluation Metrics

The Multi-Class Random Forest Model was trained on an 80/20 stratified test split (984 test cases) from `data/symptoms_disease.csv`:

| Metric | Result |
| :--- | :--- |
| **Model Type** | Multi-Class Random Forest (100 Decision Trees) |
| **Input Features** | 132 Observable Binary Symptoms |
| **Target Classes** | 41 Distinct Human Diseases |
| **Test Accuracy** | **100.00%** |
| **Weighted Precision** | **100.00%** |
| **Weighted Recall** | **100.00%** |
| **Weighted F1-Score** | **100.00%** |

---

## 6. How to Demonstrate in a College Viva (Quick-Demo Presets)

1. Open **"Disease Prediction"** (`/predict`).
2. Click any of the **Quick-Demo Preset buttons** at the top:
   - Click **"Malaria"** &rarr; Selects *chills, high fever, sweating, headache, vomiting, muscle pain*. Click **Run Diagnosis** &rarr; Predicts **Malaria** (Department: *General Medicine*).
   - Click **"GERD / Acid Reflux"** &rarr; Selects *stomach pain, acidity, ulcers on tongue, vomiting, chest pain*. Click **Run Diagnosis** &rarr; Predicts **GERD** (Department: *Gastroenterology*).
   - Click **"Dengue"** &rarr; Selects *high fever, skin rash, joint pain, chills, headache*. Click **Run Diagnosis** &rarr; Predicts **Dengue** (Department: *General Medicine*).
   - Click **"Common Cold"** &rarr; Selects *continuous sneezing, chills, fatigue, cough, headache, runny nose*. Click **Run Diagnosis** &rarr; Predicts **Common Cold** (Department: *General Medicine*).
3. On the result page, click **"Book Appointment with this Specialist"** &rarr; Directly opens the appointment desk with the doctor's department pre-filled!

---

## 7. College Viva Guide: Why this is a Multi-Class Model

- **Q: What is the difference between single-disease and multi-disease prediction?**
  *A: A single-disease model performs binary classification (Yes/No) using continuous lab numbers (e.g. glucose). A multi-disease model performs multi-class classification across 41 conditions simultaneously using a multi-hot binary vector of 132 observable clinical symptoms.*
- **Q: Why Random Forest?**
  *A: Random Forests are robust against high-dimensional binary feature spaces, avoid overfitting, and provide posterior probability distributions (`predict_proba()`) to rank differential diagnoses.*
- **Q: Why are differential diagnoses important?**
  *A: Many diseases share overlapping symptoms (e.g. fever and headache in Malaria, Dengue, and Typhoid). Outputting the top 3 differential probabilities reflects realistic medical triaging.*

---

## 8. Limitations & Safety Disclaimer

> **Educational Prototype Only**: MediCare ML was designed as an educational university computer science project. It is not an FDA-cleared diagnostic device, nor is it intended for clinical diagnosis. Always consult certified medical professionals.
