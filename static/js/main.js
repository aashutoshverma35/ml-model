/**
 * MediCare ML — Client-Side Helpers
 * Preset loading, form helpers, and sidebar toggle
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Mobile Sidebar Toggle
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebar = document.querySelector(".sidebar");
  const backdrop = document.querySelector(".sidebar-backdrop");

  if (sidebarToggle && sidebar && backdrop) {
    sidebarToggle.addEventListener("click", function () {
      sidebar.classList.toggle("show");
      backdrop.classList.toggle("show");
    });

    backdrop.addEventListener("click", function () {
      sidebar.classList.remove("show");
      backdrop.classList.remove("show");
    });
  }

  // 2. Predict Form Preset Buttons
  const presetLowRiskBtn = document.getElementById("presetLowRisk");
  const presetHighRiskBtn = document.getElementById("presetHighRisk");
  const presetClearBtn = document.getElementById("presetClear");

  if (presetLowRiskBtn) {
    presetLowRiskBtn.addEventListener("click", function () {
      setFieldValue("pregnancies", 1);
      setFieldValue("glucose", 92);
      setFieldValue("blood_pressure", 68);
      setFieldValue("skin_thickness", 20);
      setFieldValue("insulin", 75);
      setFieldValue("bmi", 22.5);
      setFieldValue("dpf", 0.245);
      setFieldValue("age", 28);
    });
  }

  if (presetHighRiskBtn) {
    presetHighRiskBtn.addEventListener("click", function () {
      setFieldValue("pregnancies", 6);
      setFieldValue("glucose", 168);
      setFieldValue("blood_pressure", 78);
      setFieldValue("skin_thickness", 35);
      setFieldValue("insulin", 180);
      setFieldValue("bmi", 34.8);
      setFieldValue("dpf", 0.655);
      setFieldValue("age", 52);
    });
  }

  if (presetClearBtn) {
    presetClearBtn.addEventListener("click", function () {
      [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "dpf",
        "age",
        "notes"
      ].forEach(function (id) {
        setFieldValue(id, "");
      });
    });
  }

  function setFieldValue(id, value) {
    const el = document.getElementById(id);
    if (el) {
      el.value = value;
      el.classList.add("is-valid");
      setTimeout(() => el.classList.remove("is-valid"), 800);
    }
  }

  // Auto dismiss bootstrap alerts after 6 seconds
  const alerts = document.querySelectorAll(".alert-dismissible");
  alerts.forEach(function (alert) {
    setTimeout(function () {
      try {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
      } catch (e) {}
    }, 6000);
  });
});
