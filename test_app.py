"""
test_app.py
===========
Automated test suite for MediCare ML Multi-Disease Symptom Prediction System.
Tests routing, database persistence, symptom validation, multi-disease
classification, and error handling.
"""

import unittest
from app import app
import database


class MediCareMLTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test client and configure test mode."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        with app.app_context():
            database.init_db()

    def test_01_dashboard_page(self):
        """Test dashboard loads with HTTP 200 and displays multi-disease metrics."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MediCare ML", response.data)
        self.assertIn(b"Total Patients", response.data)
        self.assertIn(b"Total Screenings", response.data)
        print("[PASS] Test 1: Dashboard page loads successfully.")

    def test_02_patient_management_crud(self):
        """Test creating, viewing, editing, and deleting a patient."""
        # 1. Create patient
        post_data = {
            "name": "Unit Test Patient",
            "age": "45",
            "gender": "Female",
            "phone": "+1-555-9988",
            "email": "unittest@example.com",
            "blood_group": "B+"
        }
        res_create = self.client.post("/patients/new", data=post_data, follow_redirects=True)
        self.assertEqual(res_create.status_code, 200)
        self.assertIn(b"Unit Test Patient", res_create.data)

        # Retrieve patient ID from database
        conn = database.get_db_connection()
        patient = conn.execute("SELECT * FROM patients WHERE name = ?", ("Unit Test Patient",)).fetchone()
        self.assertIsNotNone(patient)
        patient_id = patient["id"]
        conn.close()

        # 2. View details
        res_view = self.client.get(f"/patients/{patient_id}")
        self.assertEqual(res_view.status_code, 200)
        self.assertIn(b"Unit Test Patient", res_view.data)

        # 3. Edit patient
        edit_data = {
            "name": "Unit Test Patient Updated",
            "age": "46",
            "gender": "Female",
            "phone": "+1-555-9988",
            "email": "updated@example.com",
            "blood_group": "B+"
        }
        res_edit = self.client.post(f"/patients/{patient_id}/edit", data=edit_data, follow_redirects=True)
        self.assertEqual(res_edit.status_code, 200)
        self.assertIn(b"Unit Test Patient Updated", res_edit.data)

        # 4. Delete patient
        res_del = self.client.post(f"/patients/{patient_id}/delete", follow_redirects=True)
        self.assertEqual(res_del.status_code, 200)
        self.assertIn(b"has been removed", res_del.data)

        conn = database.get_db_connection()
        deleted_patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        self.assertIsNone(deleted_patient)
        conn.close()
        print("[PASS] Test 2: Patient CRUD operations verified.")

    def test_03_appointment_crud(self):
        """Test scheduling, editing, status update, and deleting appointments."""
        # 1. Book appointment
        apt_data = {
            "patient_name": "Test Appointment Subject",
            "doctor_name": "Dr. Evelyn Reed",
            "department": "General Medicine",
            "appointment_date": "2026-11-20",
            "appointment_time": "11:30 AM",
            "reason": "Follow-up for seasonal fever"
        }
        res_create = self.client.post("/appointments/new", data=apt_data, follow_redirects=True)
        self.assertEqual(res_create.status_code, 200)
        self.assertIn(b"Test Appointment Subject", res_create.data)

        conn = database.get_db_connection()
        apt = conn.execute("SELECT * FROM appointments WHERE patient_name = ?", ("Test Appointment Subject",)).fetchone()
        self.assertIsNotNone(apt)
        apt_id = apt["id"]
        conn.close()

        # 2. Update status to Completed
        res_status = self.client.post(f"/appointments/{apt_id}/status", data={"status": "Completed"}, follow_redirects=True)
        self.assertEqual(res_status.status_code, 200)
        self.assertIn(b"Completed", res_status.data)

        # 3. Delete appointment
        res_del = self.client.post(f"/appointments/{apt_id}/delete", follow_redirects=True)
        self.assertEqual(res_del.status_code, 200)
        print("[PASS] Test 3: Appointment scheduling and status updates verified.")

    def test_04_prediction_validation_errors(self):
        """Test that submitting fewer than 2 symptoms triggers a validation warning."""
        invalid_data = {
            "patient_name": "Validation Subject",
            "symptoms": ["itching"]  # only 1 symptom
        }
        response = self.client.post("/predict", data=invalid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please select at least 2 distinct symptoms", response.data)
        print("[PASS] Test 4: Boundary validation rejects insufficient symptom count.")

    def test_05_prediction_inference_malaria(self):
        """Test that submitting Malaria symptoms correctly predicts Malaria."""
        malaria_data = {
            "patient_name": "Malaria Test Patient",
            "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "muscle_pain"]
        }
        response = self.client.post("/predict", data=malaria_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Malaria", response.data)
        self.assertIn(b"General Medicine", response.data)
        print("[PASS] Test 5: ML model accurately predicts Malaria from symptom cluster.")

    def test_06_prediction_inference_gerd(self):
        """Test that submitting GERD symptoms correctly predicts GERD."""
        gerd_data = {
            "patient_name": "GERD Test Patient",
            "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]
        }
        response = self.client.post("/predict", data=gerd_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"GERD", response.data)
        self.assertIn(b"Gastroenterology", response.data)
        print("[PASS] Test 6: ML model accurately predicts GERD from digestive symptoms.")

    def test_07_prediction_inference_common_cold(self):
        """Test that submitting Common Cold symptoms predicts Common Cold."""
        cold_data = {
            "patient_name": "Cold Test Patient",
            "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose"]
        }
        response = self.client.post("/predict", data=cold_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Common Cold", response.data)
        print("[PASS] Test 7: ML model accurately predicts Common Cold from respiratory symptoms.")

    def test_08_prediction_history_and_result_page(self):
        """Test history listing and individual result rendering."""
        res_history = self.client.get("/history")
        self.assertEqual(res_history.status_code, 200)
        self.assertIn(b"Historical Symptom Screenings", res_history.data)

        # Retrieve a prediction record
        conn = database.get_db_connection()
        record = conn.execute("SELECT id FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        self.assertIsNotNone(record)
        res_result = self.client.get(f"/result/{record['id']}")
        self.assertEqual(res_result.status_code, 200)
        self.assertIn(b"Educational Prototype Notice", res_result.data)
        print("[PASS] Test 8: History and result pages render correctly.")

    def test_09_about_page_and_404_handler(self):
        """Test about page and custom 404 handler."""
        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"Multi-Disease", res_about.data)

        # Test 404
        res_404 = self.client.get("/non-existent-endpoint-test")
        self.assertEqual(res_404.status_code, 404)
        self.assertIn(b"Page Not Found", res_404.data)
        print("[PASS] Test 9: About documentation and custom 404 error handler verified.")


if __name__ == "__main__":
    unittest.main()
