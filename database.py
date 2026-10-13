"""
database.py
===========
SQLite database access and initialization layer for MediCare ML.
Stores patient registrations, clinical appointments, and multi-disease
symptom screening records with parameterized queries.
"""

import os
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "medicare.db")


def get_db_connection():
    """Establish and return a SQLite connection with Row factory enabled."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset_predictions=False):
    """Create database tables if they do not already exist and seed demo records."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT,
            blood_group TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Appointments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            patient_name TEXT NOT NULL,
            doctor_name TEXT NOT NULL,
            department TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
        )
    """)

    # Check if predictions table needs upgrade or recreation
    if reset_predictions:
        cursor.execute("DROP TABLE IF EXISTS predictions")

    # 3. Predictions table (Multi-Disease Symptom Intelligence)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            patient_name TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            severity TEXT NOT NULL,
            department TEXT NOT NULL,
            symptoms_summary TEXT NOT NULL,
            symptom_count INTEGER NOT NULL,
            differentials_json TEXT,
            precautions_json TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
        )
    """)

    conn.commit()

    # Seed initial demonstration patients if empty
    cursor.execute("SELECT COUNT(*) AS count FROM patients")
    if cursor.fetchone()["count"] == 0:
        seed_demonstration_patients(cursor)
        seed_demonstration_appointments(cursor)
        seed_demonstration_predictions(cursor)
        conn.commit()
    else:
        # If patients exist but predictions empty, seed multi-disease demo predictions
        cursor.execute("SELECT COUNT(*) AS count FROM predictions")
        if cursor.fetchone()["count"] == 0:
            seed_demonstration_predictions(cursor)
            conn.commit()

    conn.close()


def seed_demonstration_patients(cursor):
    """Seed fictional patient profiles for testing."""
    patients = [
        ("Sarah Jenkins (Demo)", 38, "Female", "+1-555-0143", "sarah.j@example.com", "O+"),
        ("Marcus Vance (Demo)", 54, "Male", "+1-555-0188", "marcus.v@example.com", "A+"),
        ("Elena Rostova (Demo)", 29, "Female", "+1-555-0129", "elena.r@example.com", "B-"),
        ("David Miller (Demo)", 62, "Male", "+1-555-0192", "david.m@example.com", "AB+")
    ]
    cursor.executemany("""
        INSERT INTO patients (name, age, gender, phone, email, blood_group)
        VALUES (?, ?, ?, ?, ?, ?)
    """, patients)


def seed_demonstration_appointments(cursor):
    """Seed fictional clinical appointments."""
    appointments = [
        (1, "Sarah Jenkins (Demo)", "Dr. Evelyn Reed", "General Medicine", "2026-10-18", "10:30 AM", "Follow-up for seasonal fever & chills", "Scheduled"),
        (2, "Marcus Vance (Demo)", "Dr. Arthur Pendelton", "Gastroenterology", "2026-10-19", "02:15 PM", "Acid reflux and stomach discomfort review", "Scheduled"),
        (3, "Elena Rostova (Demo)", "Dr. Evelyn Reed", "General Medicine", "2026-10-14", "11:00 AM", "Routine preventive health consultation", "Completed"),
        (4, "David Miller (Demo)", "Dr. Sophia Chen", "Pulmonology", "2026-10-12", "09:00 AM", "Persistent dry cough and mild wheezing", "Completed")
    ]
    cursor.executemany("""
        INSERT INTO appointments (patient_id, patient_name, doctor_name, department, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, appointments)


def seed_demonstration_predictions(cursor):
    """Seed diverse multi-disease demonstration screenings."""
    demo_records = [
        (
            1, "Sarah Jenkins (Demo)", "Malaria", 88.5, "High Attention", "General Medicine",
            "High Fever, Chills, Shaking, Vomiting, Headache, Muscle Pain", 6,
            json.dumps([{"disease": "Malaria", "probability": 88.5}, {"disease": "Dengue", "probability": 7.5}, {"disease": "Typhoid", "probability": 4.0}]),
            json.dumps(["Get a blood smear test immediately", "Complete full course of antimalarials", "Stay hydrated and rest", "Use mosquito repellents"]),
            "Demonstration screening with classic cyclic high fever and chills."
        ),
        (
            2, "Marcus Vance (Demo)", "GERD", 92.0, "Moderate", "Gastroenterology",
            "Stomach Pain, Acidity, Ulcers On Tongue, Vomiting, Chest Pain", 5,
            json.dumps([{"disease": "GERD", "probability": 92.0}, {"disease": "Peptic ulcer diseae", "probability": 5.0}, {"disease": "Heart attack", "probability": 3.0}]),
            json.dumps(["Avoid spicy and acidic foods", "Do not lie down right after meals", "Elevate bed head when sleeping", "Eat smaller meals"]),
            "Demonstration evaluation with post-meal retrosternal burning and acidity."
        ),
        (
            3, "Elena Rostova (Demo)", "Common Cold", 95.0, "Mild", "General Medicine",
            "Continuous Sneezing, Chills, Fatigue, Cough, Headache, Runny Nose", 6,
            json.dumps([{"disease": "Common Cold", "probability": 95.0}, {"disease": "Allergy", "probability": 3.5}, {"disease": "Pneumonia", "probability": 1.5}]),
            json.dumps(["Rest and stay warm", "Drink warm fluids and soups", "Use saline nasal spray", "Monitor body temperature"]),
            "Demonstration assessment presenting classic upper respiratory viral cold."
        )
    ]
    cursor.executemany("""
        INSERT INTO predictions (
            patient_id, patient_name, predicted_disease, confidence_pct, severity,
            department, symptoms_summary, symptom_count, differentials_json,
            precautions_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, demo_records)
