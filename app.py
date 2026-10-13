"""
app.py
======
MediCare ML — Multi-Disease Symptom-Based Health Prediction System.
Flask web application controller handling symptom parsing, multi-class
disease prediction, patient management, appointments, and dashboard reporting.
"""

import os
import json
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    abort
)

import database
import ml_model

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "medicare-ml-insecure-dev-key-college-demo")

# Ensure database is initialized upon startup
with app.app_context():
    database.init_db()


# -----------------------------------------------------------------------------
# Template Filters & Context Processors
# -----------------------------------------------------------------------------
@app.context_processor
def inject_global_data():
    """Inject current year and app metadata into all templates."""
    return {
        "current_year": datetime.now().year,
        "app_name": "MediCare ML"
    }


# -----------------------------------------------------------------------------
# Dashboard Route
# -----------------------------------------------------------------------------
@app.route("/")
def dashboard():
    """Render main executive healthcare dashboard."""
    conn = database.get_db_connection()

    # Query core statistics from SQLite
    total_patients = conn.execute("SELECT COUNT(*) AS c FROM patients").fetchone()["c"]
    total_appointments = conn.execute("SELECT COUNT(*) AS c FROM appointments").fetchone()["c"]
    total_predictions = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    high_attention_cases = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions WHERE severity = 'High Attention'"
    ).fetchone()["c"]

    # Recent appointments
    recent_appointments = conn.execute("""
        SELECT id, patient_name, doctor_name, department, appointment_date, appointment_time, status
        FROM appointments
        ORDER BY created_at DESC
        LIMIT 5
    """).fetchall()

    # Recent disease predictions
    recent_predictions = conn.execute("""
        SELECT id, patient_name, predicted_disease, confidence_pct, severity, department, symptoms_summary, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 5
    """).fetchall()

    # Evaluation metrics
    eval_metrics = ml_model.load_evaluation_metrics()

    conn.close()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_appointments=total_appointments,
        total_predictions=total_predictions,
        high_attention_cases=high_attention_cases,
        recent_appointments=recent_appointments,
        recent_predictions=recent_predictions,
        eval_metrics=eval_metrics
    )


# -----------------------------------------------------------------------------
# Patient Management Routes
# -----------------------------------------------------------------------------
@app.route("/patients")
def patients_list():
    """List registered patients with optional search query."""
    search_query = request.args.get("q", "").strip()
    conn = database.get_db_connection()

    if search_query:
        wildcard = f"%{search_query}%"
        patients = conn.execute("""
            SELECT * FROM patients
            WHERE name LIKE ? OR phone LIKE ? OR email LIKE ?
            ORDER BY created_at DESC
        """, (wildcard, wildcard, wildcard)).fetchall()
    else:
        patients = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()

    conn.close()
    return render_template("patients.html", patients=patients, search_query=search_query)


@app.route("/patients/new", methods=["GET", "POST"])
def patient_new():
    """Register a new patient record."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age_raw = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        blood_group = request.form.get("blood_group", "").strip()

        errors = []
        if not name:
            errors.append("Patient full name is required.")
        if not age_raw:
            errors.append("Patient age is required.")
        else:
            try:
                age = int(age_raw)
                if age < 1 or age > 120:
                    errors.append("Patient age must be between 1 and 120.")
            except ValueError:
                errors.append("Age must be a valid integer.")
                age = None

        if gender not in ["Male", "Female", "Other"]:
            errors.append("Please select a valid gender option.")
        if not phone:
            errors.append("Contact phone number is required.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("patient_form.html", patient=request.form, action="New")

        conn = database.get_db_connection()
        conn.execute("""
            INSERT INTO patients (name, age, gender, phone, email, blood_group)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, age, gender, phone, email, blood_group))
        conn.commit()
        conn.close()

        flash(f"Patient record for '{name}' created successfully.", "success")
        return redirect(url_for("patients_list"))

    return render_template("patient_form.html", patient={}, action="New")


@app.route("/patients/<int:patient_id>")
def patient_detail(patient_id):
    """View details for an individual patient, including appointments and predictions."""
    conn = database.get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        conn.close()
        abort(404)

    appointments = conn.execute("""
        SELECT * FROM appointments WHERE patient_id = ? ORDER BY appointment_date DESC
    """, (patient_id,)).fetchall()

    predictions = conn.execute("""
        SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC
    """, (patient_id,)).fetchall()

    conn.close()
    return render_template(
        "patient_detail.html",
        patient=patient,
        appointments=appointments,
        predictions=predictions
    )


@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
def patient_edit(patient_id):
    """Edit an existing patient's details."""
    conn = database.get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        conn.close()
        abort(404)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age_raw = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        blood_group = request.form.get("blood_group", "").strip()

        errors = []
        if not name:
            errors.append("Patient full name is required.")
        if not age_raw:
            errors.append("Patient age is required.")
        else:
            try:
                age = int(age_raw)
                if age < 1 or age > 120:
                    errors.append("Patient age must be between 1 and 120.")
            except ValueError:
                errors.append("Age must be a valid integer.")
                age = None

        if gender not in ["Male", "Female", "Other"]:
            errors.append("Please select a valid gender option.")
        if not phone:
            errors.append("Contact phone number is required.")

        if errors:
            for err in errors:
                flash(err, "danger")
            conn.close()
            return render_template("patient_form.html", patient=request.form, action="Edit")

        conn.execute("""
            UPDATE patients
            SET name = ?, age = ?, gender = ?, phone = ?, email = ?, blood_group = ?
            WHERE id = ?
        """, (name, age, gender, phone, email, blood_group, patient_id))
        conn.commit()
        conn.close()

        flash(f"Patient record for '{name}' updated successfully.", "success")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    conn.close()
    return render_template("patient_form.html", patient=patient, action="Edit")


@app.route("/patients/<int:patient_id>/delete", methods=["POST"])
def patient_delete(patient_id):
    """Delete a patient record with confirmation."""
    conn = database.get_db_connection()
    patient = conn.execute("SELECT name FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        conn.close()
        abort(404)

    conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()

    flash(f"Patient record for '{patient['name']}' has been removed.", "info")
    return redirect(url_for("patients_list"))


# -----------------------------------------------------------------------------
# Multi-Disease Symptom Prediction Routes (ML Core)
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Symptom-based multi-disease screening and prediction endpoint."""
    conn = database.get_db_connection()
    patients = conn.execute("SELECT id, name, age FROM patients ORDER BY name ASC").fetchall()
    conn.close()

    all_symptoms = ml_model.get_all_symptoms()
    symptom_categories = ml_model.SYMPTOM_CATEGORIES
    demo_presets = ml_model.DEMO_PRESETS

    if request.method == "POST":
        # Patient association
        patient_id_raw = request.form.get("patient_id", "").strip()
        custom_patient_name = request.form.get("patient_name", "").strip()

        patient_id = None
        patient_name = custom_patient_name or "Anonymous Patient"

        if patient_id_raw:
            try:
                patient_id = int(patient_id_raw)
                conn = database.get_db_connection()
                matched = conn.execute("SELECT name FROM patients WHERE id = ?", (patient_id,)).fetchone()
                conn.close()
                if matched:
                    patient_name = matched["name"]
            except ValueError:
                patient_id = None

        # Extract selected symptoms from checkboxes / multi-select
        selected_symptoms = request.form.getlist("symptoms")
        # Also check if passed via comma-separated string from preset helper
        symptoms_str = request.form.get("symptoms_raw", "").strip()
        if symptoms_str:
            extra_symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
            selected_symptoms = list(set(selected_symptoms + extra_symptoms))

        if not selected_symptoms or len(selected_symptoms) < 2:
            flash("Please select at least 2 distinct symptoms so the ML model can evaluate differential probabilities.", "warning")
            return render_template(
                "predict.html",
                patients=patients,
                all_symptoms=all_symptoms,
                categories=symptom_categories,
                presets=demo_presets,
                selected_symptoms=selected_symptoms,
                patient_id=patient_id_raw,
                patient_name=custom_patient_name
            )

        # Run Multi-Disease Inference through Random Forest Classifier
        try:
            result = ml_model.predict_disease_from_symptoms(selected_symptoms)
        except Exception as e:
            flash(f"Inference Error: {str(e)}", "danger")
            return render_template(
                "predict.html",
                patients=patients,
                all_symptoms=all_symptoms,
                categories=symptom_categories,
                presets=demo_presets,
                selected_symptoms=selected_symptoms,
                patient_id=patient_id_raw,
                patient_name=custom_patient_name
            )

        # Insert prediction record into SQLite
        conn = database.get_db_connection()
        cursor = conn.cursor()
        symptoms_summary = ", ".join(result["selected_symptoms"])
        differentials_json = json.dumps(result["differentials"])
        precautions_json = json.dumps(result["precautions"])

        cursor.execute("""
            INSERT INTO predictions (
                patient_id, patient_name, predicted_disease, confidence_pct,
                severity, department, symptoms_summary, symptom_count,
                differentials_json, precautions_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            patient_name,
            result["predicted_disease"],
            result["confidence_pct"],
            result["severity"],
            result["department"],
            symptoms_summary,
            result["symptom_count"],
            differentials_json,
            precautions_json,
            request.form.get("notes", "").strip()
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        flash(f"Symptom cluster evaluated: Likely match '{result['predicted_disease']}'.", "success")
        return redirect(url_for("prediction_result", record_id=record_id))

    # GET Request: render symptom selection form
    selected_patient_id = request.args.get("patient_id")
    prefill_patient_name = ""
    if selected_patient_id:
        try:
            pid = int(selected_patient_id)
            conn = database.get_db_connection()
            p = conn.execute("SELECT name FROM patients WHERE id = ?", (pid,)).fetchone()
            conn.close()
            if p:
                prefill_patient_name = p["name"]
        except Exception:
            pass

    return render_template(
        "predict.html",
        patients=patients,
        all_symptoms=all_symptoms,
        categories=symptom_categories,
        presets=demo_presets,
        selected_symptoms=[],
        patient_id=selected_patient_id or "",
        patient_name=prefill_patient_name
    )


@app.route("/result/<int:record_id>")
def prediction_result(record_id):
    """Display comprehensive multi-disease prediction results."""
    conn = database.get_db_connection()
    record = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    conn.close()

    if not record:
        abort(404)

    # Parse JSON fields
    differentials = []
    if record["differentials_json"]:
        try:
            differentials = json.loads(record["differentials_json"])
        except Exception:
            pass

    precautions = []
    if record["precautions_json"]:
        try:
            precautions = json.loads(record["precautions_json"])
        except Exception:
            pass

    # Disease explanation
    disease_meta = ml_model.DISEASE_METADATA.get(record["predicted_disease"], {
        "description": "A clinical condition identified from the patient's submitted symptom cluster.",
        "department": record["department"],
        "severity": record["severity"]
    })

    # Reported symptoms list
    reported_symptoms = [s.strip() for s in record["symptoms_summary"].split(",") if s.strip()]

    return render_template(
        "result.html",
        record=record,
        disease_meta=disease_meta,
        differentials=differentials,
        precautions=precautions,
        reported_symptoms=reported_symptoms
    )


# -----------------------------------------------------------------------------
# Appointment Management Routes
# -----------------------------------------------------------------------------
@app.route("/appointments")
def appointments_list():
    """List clinical appointments with status filters."""
    status_filter = request.args.get("status", "").strip()
    conn = database.get_db_connection()

    if status_filter and status_filter in ["Scheduled", "Completed", "Cancelled"]:
        appointments = conn.execute("""
            SELECT * FROM appointments WHERE status = ? ORDER BY appointment_date ASC, appointment_time ASC
        """, (status_filter,)).fetchall()
    else:
        appointments = conn.execute("""
            SELECT * FROM appointments ORDER BY appointment_date ASC, appointment_time ASC
        """).fetchall()

    conn.close()
    return render_template("appointments.html", appointments=appointments, status_filter=status_filter)


@app.route("/appointments/new", methods=["GET", "POST"])
def appointment_new():
    """Book a new clinical appointment (pre-fills department if referred)."""
    conn = database.get_db_connection()
    patients = conn.execute("SELECT id, name FROM patients ORDER BY name ASC").fetchall()

    if request.method == "POST":
        patient_id_raw = request.form.get("patient_id", "").strip()
        custom_patient_name = request.form.get("patient_name", "").strip()
        doctor_name = request.form.get("doctor_name", "").strip()
        department = request.form.get("department", "").strip()
        appointment_date = request.form.get("appointment_date", "").strip()
        appointment_time = request.form.get("appointment_time", "").strip()
        reason = request.form.get("reason", "").strip()

        errors = []
        patient_id = None
        patient_name = custom_patient_name

        if patient_id_raw:
            try:
                patient_id = int(patient_id_raw)
                matched = conn.execute("SELECT name FROM patients WHERE id = ?", (patient_id,)).fetchone()
                if matched:
                    patient_name = matched["name"]
            except ValueError:
                patient_id = None

        if not patient_name:
            errors.append("Patient name or selection is required.")
        if not doctor_name:
            errors.append("Attending doctor name is required.")
        if not department:
            errors.append("Department is required.")
        if not appointment_date:
            errors.append("Appointment date is required.")
        if not appointment_time:
            errors.append("Appointment time is required.")
        if not reason:
            errors.append("Reason for appointment is required.")

        if errors:
            for err in errors:
                flash(err, "danger")
            conn.close()
            return render_template("appointment_form.html", patients=patients, appointment=request.form, action="New")

        conn.execute("""
            INSERT INTO appointments (
                patient_id, patient_name, doctor_name, department,
                appointment_date, appointment_time, reason, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Scheduled')
        """, (patient_id, patient_name, doctor_name, department, appointment_date, appointment_time, reason))
        conn.commit()
        conn.close()

        flash("Appointment booked successfully.", "success")
        return redirect(url_for("appointments_list"))

    # GET request pre-fill
    prefill_dept = request.args.get("dept", "General Medicine")
    prefill_reason = request.args.get("reason", "")
    prefill_patient_id = request.args.get("patient_id", "")
    prefill_patient_name = request.args.get("patient_name", "")

    conn.close()
    return render_template(
        "appointment_form.html",
        patients=patients,
        appointment={
            "department": prefill_dept,
            "reason": prefill_reason,
            "patient_id": prefill_patient_id,
            "patient_name": prefill_patient_name
        },
        action="New"
    )


@app.route("/appointments/<int:appointment_id>/edit", methods=["GET", "POST"])
def appointment_edit(appointment_id):
    """Edit an existing appointment."""
    conn = database.get_db_connection()
    appointment = conn.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,)).fetchone()
    patients = conn.execute("SELECT id, name FROM patients ORDER BY name ASC").fetchall()

    if not appointment:
        conn.close()
        abort(404)

    if request.method == "POST":
        doctor_name = request.form.get("doctor_name", "").strip()
        department = request.form.get("department", "").strip()
        appointment_date = request.form.get("appointment_date", "").strip()
        appointment_time = request.form.get("appointment_time", "").strip()
        reason = request.form.get("reason", "").strip()
        status = request.form.get("status", "Scheduled").strip()

        errors = []
        if not doctor_name:
            errors.append("Doctor name is required.")
        if not department:
            errors.append("Department is required.")
        if not appointment_date:
            errors.append("Appointment date is required.")
        if not appointment_time:
            errors.append("Appointment time is required.")
        if not reason:
            errors.append("Reason is required.")
        if status not in ["Scheduled", "Completed", "Cancelled"]:
            errors.append("Invalid status selected.")

        if errors:
            for err in errors:
                flash(err, "danger")
            conn.close()
            return render_template(
                "appointment_form.html",
                patients=patients,
                appointment=request.form,
                action="Edit"
            )

        conn.execute("""
            UPDATE appointments
            SET doctor_name = ?, department = ?, appointment_date = ?,
                appointment_time = ?, reason = ?, status = ?
            WHERE id = ?
        """, (doctor_name, department, appointment_date, appointment_time, reason, status, appointment_id))
        conn.commit()
        conn.close()

        flash("Appointment updated successfully.", "success")
        return redirect(url_for("appointments_list"))

    conn.close()
    return render_template(
        "appointment_form.html",
        patients=patients,
        appointment=appointment,
        action="Edit"
    )


@app.route("/appointments/<int:appointment_id>/status", methods=["POST"])
def appointment_update_status(appointment_id):
    """Quickly update appointment status."""
    new_status = request.form.get("status", "").strip()
    if new_status in ["Scheduled", "Completed", "Cancelled"]:
        conn = database.get_db_connection()
        conn.execute("UPDATE appointments SET status = ? WHERE id = ?", (new_status, appointment_id))
        conn.commit()
        conn.close()
        flash(f"Appointment status updated to '{new_status}'.", "info")
    return redirect(url_for("appointments_list"))


@app.route("/appointments/<int:appointment_id>/delete", methods=["POST"])
def appointment_delete(appointment_id):
    """Delete an appointment."""
    conn = database.get_db_connection()
    conn.execute("DELETE FROM appointments WHERE id = ?", (appointment_id,))
    conn.commit()
    conn.close()
    flash("Appointment removed successfully.", "info")
    return redirect(url_for("appointments_list"))


# -----------------------------------------------------------------------------
# Prediction History Route
# -----------------------------------------------------------------------------
@app.route("/history")
def prediction_history():
    """List historical multi-disease prediction records."""
    query = request.args.get("q", "").strip()
    disease_filter = request.args.get("disease", "").strip()

    conn = database.get_db_connection()

    sql = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if query:
        sql += " AND (patient_name LIKE ? OR symptoms_summary LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])

    if disease_filter:
        sql += " AND predicted_disease = ?"
        params.append(disease_filter)

    sql += " ORDER BY created_at DESC"

    records = conn.execute(sql, params).fetchall()

    # List of all distinct diseases in predictions for filter dropdown
    distinct_diseases = [r[0] for r in conn.execute(
        "SELECT DISTINCT predicted_disease FROM predictions ORDER BY predicted_disease ASC"
    ).fetchall()]

    conn.close()

    return render_template(
        "history.html",
        records=records,
        search_query=query,
        disease_filter=disease_filter,
        distinct_diseases=distinct_diseases
    )


# -----------------------------------------------------------------------------
# About & ML Architecture Route
# -----------------------------------------------------------------------------
@app.route("/about")
def about():
    """Project documentation, multi-disease architecture, and viva guide."""
    eval_metrics = ml_model.load_evaluation_metrics()
    return render_template(
        "about.html",
        eval_metrics=eval_metrics,
        disease_metadata=ml_model.DISEASE_METADATA,
        symptom_categories=ml_model.SYMPTOM_CATEGORIES
    )


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "error.html",
        error_code=404,
        error_title="Page Not Found",
        error_message="The requested medical resource or patient record could not be found."
    ), 404


@app.errorhandler(500)
def server_error(e):
    return render_template(
        "error.html",
        error_code=500,
        error_title="Internal Server Error",
        error_message="An unexpected server error occurred while processing medical records."
    ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting MediCare ML server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
