# app.py
import os
import io
import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash

import numpy as np
import pandas as pd
import joblib

# utils must export these functions/objects
from utils import (
    input_preprocessing,
    keep_numeric,
    chart_save,
    sample,
    generate_insights,
    hesapla,
)

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_artifacts" / "model.sav"
SCALE_PATH = BASE_DIR / "model_artifacts" / "scale.sav"
PREDICTIONS_DB = BASE_DIR / "predictions.json"
CHARTS_DIR = BASE_DIR / "static" / "charts"

os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------- Flask ----------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace-with-a-secure-random-key"

# ---------- load model & scaler ----------
def safe_load(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

model = safe_load(MODEL_PATH)
scaler = safe_load(SCALE_PATH)

# ---------- persistence ----------
def append_prediction_record(record: dict):
    if PREDICTIONS_DB.exists():
        with open(PREDICTIONS_DB, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.insert(0, record)
    data = data[:500]
    with open(PREDICTIONS_DB, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_prediction_history():
    if PREDICTIONS_DB.exists():
        return json.load(open(PREDICTIONS_DB, "r"))
    return []

# ---------- session helpers ----------
def get_form_data():
    return session.get("form_data", {})

def set_form_data(data):
    session["form_data"] = data
    session.modified = True

# ---------- Routes: home & multi-step form ----------
@app.route("/", methods=["GET"])
def home():
    session.setdefault("form_data", {})
    return render_template("home.html")

@app.route("/personal", methods=["GET","POST"])
def personal():
    if request.method == "POST":
        d = get_form_data()
        d.update({
            "height": request.form.get("height",""),
            "weight": request.form.get("weight",""),
            "sex": request.form.get("sex","female"),
            "diet": request.form.get("diet","omnivore"),
            "social": request.form.get("social","sometimes"),
            "shower": request.form.get("shower","daily")
        })
        set_form_data(d)
        return redirect(url_for("travel"))
    return render_template("personal.html", data=get_form_data())

@app.route("/travel", methods=["GET","POST"])
def travel():
    if request.method == "POST":
        d = get_form_data()
        d.update({
            "transport": request.form.get("transport","public"),
            "vehicle_type": request.form.get("vehicle_type","None"),
            "vehicle_km": request.form.get("vehicle_km","0"),
            "air_travel": request.form.get("air_travel","never")
        })
        set_form_data(d)
        return redirect(url_for("waste"))
    return render_template("travel.html", data=get_form_data())

@app.route("/waste", methods=["GET","POST"])
def waste():
    if request.method == "POST":
        d = get_form_data()
        recycle = request.form.getlist("recycle")
        d.update({
            "waste_bag": request.form.get("waste_bag","medium"),
            "waste_count": request.form.get("waste_count","0"),
            "recycle": recycle
        })
        set_form_data(d)
        return redirect(url_for("energy"))
    return render_template("waste.html", data=get_form_data())

@app.route("/energy", methods=["GET","POST"])
def energy():
    if request.method == "POST":
        d = get_form_data()
        cooking = request.form.getlist("for_cooking")
        d.update({
            "heating_energy": request.form.get("heating_energy","natural gas"),
            "for_cooking": cooking,
            "energy_efficiency": request.form.get("energy_efficiency","No"),
            "daily_tv_pc": request.form.get("daily_tv_pc","0"),
            "internet_daily": request.form.get("internet_daily","0")
        })
        set_form_data(d)
        return redirect(url_for("consumption"))
    return render_template("energy.html", data=get_form_data())

@app.route("/consumption", methods=["GET","POST"])
def consumption():
    if request.method == "POST":
        d = get_form_data()
        d.update({
            "grocery_bill": request.form.get("grocery_bill","0"),
            "clothes_monthly": request.form.get("clothes_monthly","0")
        })
        set_form_data(d)
        return redirect(url_for("predict"))
    return render_template("consumption.html", data=get_form_data())

# ---------- Predict (creates per-record chart, insights, components) ----------
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = get_form_data()
    if not data:
        flash("Please fill the calculator first.", "warning")
        return redirect(url_for("personal"))

    # BMI -> body type
    try:
        height = float(data.get("height",160))
        weight = float(data.get("weight",75))
        if height <= 0: height = 160.0
        if weight <= 0: weight = 75.0
    except Exception:
        height, weight = 160.0, 75.0

    calculation = weight / (height / 100) ** 2
    body_type = "underweight" if calculation < 18.5 else \
                 "normal" if calculation < 25 else \
                 "overweight" if calculation < 30 else "obese"

    # input dict (same fields as Streamlit)
    data_dict = {
        "Body Type": body_type,
        "Sex": data.get("sex","female"),
        "Diet": data.get("diet","omnivore"),
        "How Often Shower": data.get("shower","daily"),
        "Heating Energy Source": data.get("heating_energy","natural gas"),
        "Transport": data.get("transport","public"),
        "Social Activity": data.get("social","sometimes"),
        "Monthly Grocery Bill": float(data.get("grocery_bill",0)),
        "Frequency of Traveling by Air": data.get("air_travel","never"),
        "Vehicle Monthly Distance Km": float(data.get("vehicle_km",0)),
        "Waste Bag Size": data.get("waste_bag","medium"),
        "Waste Bag Weekly Count": float(data.get("waste_count",0)),
        "How Long TV PC Daily Hour": float(data.get("daily_tv_pc",0)),
        "Vehicle Type": data.get("vehicle_type","None"),
        "How Many New Clothes Monthly": float(data.get("clothes_monthly",0)),
        "How Long Internet Daily Hour": float(data.get("internet_daily",0)),
        "Energy efficiency": data.get("energy_efficiency","No"),
    }

    for c in data.get("for_cooking", []):
        data_dict[f"Cooking_with_{c}"] = 1
    for r in data.get("recycle", []):
        data_dict[f"Do You Recyle_{r}"] = 1

    df = pd.DataFrame(data_dict, index=[0])
    processed = input_preprocessing(df)
    numeric = keep_numeric(processed)

    # align with sample template
    sample_row = pd.DataFrame(data=sample, index=[0])
    sample_row[sample_row.columns] = 0
    for col in numeric.columns:
        if col not in sample_row.columns:
            sample_row[col] = 0
    sample_row[numeric.columns] = numeric.values
    sample_row = sample_row[sample_row.columns]

    # scale & prediction
    scaled = scaler.transform(sample_row)
    pred_log = model.predict(scaled)[0]
    prediction = round(np.exp(pred_log))

    # insights
    insights = generate_insights(sample_row, prediction)

    # components (Travel/Energy/Waste/Diet)
    try:
        components = hesapla(model, scaler, sample_row)
    except Exception as e:
        app.logger.exception("hesapla failed: %s", e)
        components = {"Travel": 0.0, "Energy": 0.0, "Waste": 0.0, "Diet": 0.0}

    total_comp = sum(components.values()) if components else 0.0
    comp_perc = {}
    for k, v in components.items():
        comp_perc[k] = round((v / total_comp * 100), 1) if total_comp > 0 else 0.0

    # equivalents
    tonnes_per_year = round((prediction * 12) / 1000, 2)
    car_km_equiv = int(round(prediction / 0.24)) if prediction > 0 else 0
    short_flight_equiv = int(round(prediction / 150)) if prediction > 0 else 0
    tree_count_est = round(prediction / 411.4) if prediction > 0 else 0

    # top actions (3)
    top_actions = []
    if insights and "recommendations" in insights:
        for rec in insights["recommendations"][:3]:
            top_actions.append({
                "title": rec.get("title"),
                "estimate": rec.get("estimate_kg_per_month"),
                "detail": rec.get("detail")
            })

    # create per-record chart and save
    record_id = str(uuid.uuid4())
    chart_filename = f"result-{record_id}.png"
    absolute_chart_path = CHARTS_DIR / chart_filename
    chart_rel_path = f"charts/{chart_filename}"
    try:
        chart_save(model, scaler, sample_row, prediction, str(absolute_chart_path))
    except Exception as e:
        app.logger.exception("Chart save failed: %s", e)
        chart_rel_path = None

    # save record
    rec = {
        "id": record_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": int(prediction),
        "body_type": body_type,
        "data": data_dict,
        "insights": insights,
        "chart": chart_rel_path
    }
    append_prediction_record(rec)

    # render result
    return render_template(
        "result.html",
        prediction=prediction,
        tree_count=tree_count_est,
        insights=insights,
        chart_rel_path=rec["chart"],
        components=components,
        comp_perc=comp_perc,
        tonnes_per_year=tonnes_per_year,
        car_km_equiv=car_km_equiv,
        short_flight_equiv=short_flight_equiv,
        tree_count_est=tree_count_est,
        top_actions=top_actions
    )

# ---------- Dashboard ----------
@app.route("/dashboard", methods=["GET"])
def dashboard():
    history = load_prediction_history()
    return render_template("dashboard.html", history=history)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if PREDICTIONS_DB.exists():
        PREDICTIONS_DB.unlink()
    # clean charts
    for f in CHARTS_DIR.glob("result-*.png"):
        try:
            f.unlink()
        except Exception:
            pass
    flash("History cleared.", "success")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    os.makedirs(BASE_DIR / "static" / "charts", exist_ok=True)
    app.run(debug=True)
