# utils.py
import os
import io
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sample = {
    "Body Type": 2,
    "Sex": 0,
    "How Often Shower": 1,
    "Social Activity": 2,
    "Monthly Grocery Bill": 230,
    "Frequency of Traveling by Air": 2,
    "Vehicle Monthly Distance Km": 210,
    "Waste Bag Size": 2,
    "Waste Bag Weekly Count": 4,
    "How Long TV PC Daily Hour": 7,
    "How Many New Clothes Monthly": 26,
    "How Long Internet Daily Hour": 1,
    "Energy efficiency": 0,
    "Do You Recyle_Paper": 0,
    "Do You Recyle_Plastic": 0,
    "Do You Recyle_Glass": 0,
    "Do You Recyle_Metal": 1,
    "Cooking_with_stove": 1,
    "Cooking_with_oven": 1,
    "Cooking_with_microwave": 0,
    "Cooking_with_grill": 0,
    "Cooking_with_airfryer": 1,
    "Diet_omnivore": 0,
    "Diet_pescatarian": 1,
    "Diet_vegan": 0,
    "Diet_vegetarian": 0,
    "Heating Energy Source_coal": 1,
    "Heating Energy Source_electricity": 0,
    "Heating Energy Source_natural gas": 0,
    "Heating Energy Source_wood": 0,
    "Transport_private": 0,
    "Transport_public": 1,
    "Transport_walk/bicycle": 0,
    "Vehicle Type_None": 1,
    "Vehicle Type_diesel": 0,
    "Vehicle Type_electric": 0,
    "Vehicle Type_hybrid": 0,
    "Vehicle Type_lpg": 0,
    "Vehicle Type_petrol": 0,
}

EXPECTED_RECYCLE_ITEMS = ["plastic", "paper", "metal", "glass"]
EXPECTED_COOKING_ITEMS = ["stove", "oven", "microwave", "grill", "airfryer"]

def split_to_list(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple, set)):
        return [str(x).strip().lower() for x in cell if str(x).strip()]
    text = str(cell)
    parts = re.split(r'[;,/|]', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for sub in p.split(","):
            s = sub.strip()
            if s:
                out.append(s.lower())
    return list(dict.fromkeys(out))

def input_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]

    if "Recycling" in data.columns:
        rec_series = data["Recycling"].apply(split_to_list)
        for item in EXPECTED_RECYCLE_ITEMS:
            colname = f"Do You Recyle_{item.capitalize()}"
            data[colname] = rec_series.apply(lambda lst: 1 if item in lst else 0)

    if "Cooking_With" in data.columns:
        cook_series = data["Cooking_With"].apply(split_to_list)
        for item in EXPECTED_COOKING_ITEMS:
            colname = f"Cooking_with_{item}"
            data[colname] = cook_series.apply(lambda lst: 1 if item in lst else 0)

    if "Body Type" in data.columns:
        data["Body Type"] = data["Body Type"].map({"underweight":0,"normal":1,"overweight":2,"obese":3})
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"female":0,"male":1})

    cat_cols = []
    for col in ["Diet","Heating Energy Source","Transport","Vehicle Type"]:
        if col in data.columns:
            cat_cols.append(col)
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, dtype=int)

    if "How Often Shower" in data.columns:
        data["How Often Shower"] = data["How Often Shower"].map({"less frequently":0,"daily":1,"twice a day":2,"more frequently":3})
    if "Social Activity" in data.columns:
        data["Social Activity"] = data["Social Activity"].map({"never":0,"sometimes":1,"often":2})
    if "Frequency of Traveling by Air" in data.columns:
        data["Frequency of Traveling by Air"] = data["Frequency of Traveling by Air"].map({"never":0,"rarely":1,"frequently":2,"very frequently":3})
    if "Waste Bag Size" in data.columns:
        data["Waste Bag Size"] = data["Waste Bag Size"].map({"small":0,"medium":1,"large":2,"extra large":3})
    if "Energy efficiency" in data.columns:
        data["Energy efficiency"] = data["Energy efficiency"].map({"No":0,"Sometimes":1,"Yes":2})

    data = data.fillna(0)
    return data

def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def hesapla(model, ss, sample_df):
    # Travel
    copy_df = sample_df.copy()
    travels = [
        "Frequency of Traveling by Air",
        "Vehicle Monthly Distance Km",
        "Transport_private",
        "Transport_public",
        "Transport_walk/bicycle",
        "Vehicle Type_None",
        "Vehicle Type_diesel",
        "Vehicle Type_electric",
        "Vehicle Type_hybrid",
        "Vehicle Type_lpg",
        "Vehicle Type_petrol",
    ]
    copy_df[list(set(copy_df.columns) - set(travels))] = 0
    travel = np.exp(model.predict(ss.transform(copy_df)))

    # Energy
    copy_df = sample_df.copy()
    energys = [
        "Heating Energy Source_coal",
        "How Often Shower",
        "How Long TV PC Daily Hour",
        "Heating Energy Source_electricity",
        "How Long Internet Daily Hour",
        "Heating Energy Source_natural gas",
        "Cooking_with_stove",
        "Cooking_with_oven",
        "Cooking_with_microwave",
        "Cooking_with_grill",
        "Cooking_with_airfryer",
        "Heating Energy Source_wood",
        "Energy efficiency",
    ]
    copy_df[list(set(copy_df.columns) - set(energys))] = 0
    energy = np.exp(model.predict(ss.transform(copy_df)))

    # Waste
    copy_df = sample_df.copy()
    wastes = [
        "Do You Recyle_Paper",
        "How Many New Clothes Monthly",
        "Waste Bag Size",
        "Waste Bag Weekly Count",
        "Do You Recyle_Plastic",
        "Do You Recyle_Glass",
        "Do You Recyle_Metal",
        "Social Activity",
    ]
    copy_df[list(set(copy_df.columns) - set(wastes))] = 0
    waste = np.exp(model.predict(ss.transform(copy_df)))

    # Diet
    copy_df = sample_df.copy()
    diets = [
        "Diet_omnivore",
        "Diet_pescatarian",
        "Diet_vegan",
        "Diet_vegetarian",
        "Monthly Grocery Bill",
        "Transport_private",
        "Transport_public",
        "Transport_walk/bicycle",
        "Heating Energy Source_coal",
        "Heating Energy Source_electricity",
        "Heating Energy Source_natural gas",
        "Heating Energy Source_wood",
    ]
    copy_df[list(set(copy_df.columns) - set(diets))] = 0
    diet = np.exp(model.predict(ss.transform(copy_df)))

    return {
        "Travel": float(travel[0]),
        "Energy": float(energy[0]),
        "Waste": float(waste[0]),
        "Diet": float(diet[0]),
    }

def chart_save(model, scaler, sample_df, prediction, out_path: str):
    p = hesapla(model, scaler, sample_df)
    labels = list(p.keys())
    values = [float(v) for v in p.values()]

    plt.figure(figsize=(7,4.5))
    x_pos = range(len(labels))
    bars = plt.bar(x_pos, values)
    plt.xticks(x_pos, labels, fontsize=12, weight='bold')
    plt.ylabel("kg CO₂e / month", fontsize=11)
    plt.title("Component-wise monthly emissions", fontsize=13, weight='bold')
    plt.grid(axis='y', alpha=0.25)

    max_val = max(values) if values else 0
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height + max_val * 0.02, f"{height:.0f}",
                 ha='center', va='bottom', fontsize=10)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="PNG", transparent=True, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    chart_img = Image.open(buf).convert("RGBA")

    bg_path = os.path.join("media", "default.png")
    if os.path.exists(bg_path):
        background = Image.open(bg_path).convert("RGBA")
    else:
        background = Image.new("RGBA", (900,900), (255,255,255,255))

    draw = ImageDraw.Draw(background)
    try:
        font1 = ImageFont.truetype(os.path.join("style", "ArchivoBlack-Regular.ttf"), size=36)
        font2 = ImageFont.truetype(os.path.join("style", "arialuni.ttf"), size=30)
    except Exception:
        font1 = None
        font2 = None

    if font1:
        draw.text((40,40), "How big is your\nCarbon Footprint?", font=font1, fill="#039e8e")
    if font2:
        draw.text((40,180), f"Monthly Emission\n\n  {prediction:.0f} kgCO₂e", font=font2, fill="#039e8e")

    chart_w, chart_h = chart_img.size
    bg_w, bg_h = background.size
    max_w = bg_w - 80
    max_h = bg_h - 360
    scale = min(max_w / chart_w, max_h / chart_h, 1.0)
    if scale != 1.0:
        chart_img = chart_img.resize((int(chart_w * scale), int(chart_h * scale)), Image.ANTIALIAS)
        chart_w, chart_h = chart_img.size

    paste_x = (bg_w - chart_w) // 2
    paste_y = 260
    background.paste(chart_img, (paste_x, paste_y), chart_img)

    ayak_path = os.path.join("media","ayak.png")
    if os.path.exists(ayak_path):
        ayak = Image.open(ayak_path).convert("RGBA")
        ayak = ayak.resize((int(chart_w * 0.9), int(chart_h * 0.22)), Image.ANTIALIAS)
        ax = paste_x + (chart_w - ayak.size[0]) // 2
        ay = paste_y + chart_h - int(ayak.size[1] * 0.9)
        background.paste(ayak, (ax, ay), ayak)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    background.save(out_path, format="PNG")
    return out_path

def generate_insights(sample_df, prediction):
    recs = []
    row = sample_df.iloc[0]
    def get(col, default=0):
        return float(row[col]) if col in row and pd.notna(row[col]) else default

    vehicle_km = get("Vehicle Monthly Distance Km", 0)
    transport_private = bool(get("Transport_private", 0))
    transport_public = bool(get("Transport_public", 0))
    if transport_private and vehicle_km > 0:
        saving = vehicle_km * 0.24 * 0.5
        recs.append({
            "title": "Reduce private car use",
            "detail": "Try public transport, carpooling, or switch to an efficient vehicle for some trips.",
            "estimate_kg_per_month": round(saving, 1),
            "why": "Assumed ~0.24 kgCO₂e/km and ~50% saving when shifting modes."
        })
    elif transport_public:
        recs.append({
            "title": "Good: public transport",
            "detail": "Public transport reduces per-person emissions — continue this habit.",
            "estimate_kg_per_month": 0,
            "why": ""
        })

    air_freq = int(get("Frequency of Traveling by Air", 0))
    if air_freq >= 2:
        saving = 100 * air_freq
        recs.append({
            "title": "Reduce air travel",
            "detail": "Consider fewer flights or alternatives like trains for short trips.",
            "estimate_kg_per_month": round(saving,1),
            "why": "Flights have high per-trip emissions."
        })

    energy_eff = get("Energy efficiency", 0)
    hours_tv = get("How Long TV PC Daily Hour", 0)
    if energy_eff < 2:
        recs.append({
            "title": "Increase energy efficiency",
            "detail": "LED bulbs, energy-efficient appliances, and unplugging idle devices help reduce electricity demand.",
            "estimate_kg_per_month": 20,
            "why": "Efficiency measures reduce monthly household energy."
        })
    if hours_tv > 4:
        recs.append({
            "title": "Reduce screen hours",
            "detail": "Lower TV/PC usage an hour daily or use power-saving modes.",
            "estimate_kg_per_month": round((hours_tv - 3) * 2 * 30, 1),
            "why": "Screen electricity adds to household emissions."
        })

    recycled_any = any(get(f"Do You Recyle_{c.capitalize()}", 0) for c in ["paper","plastic","glass","metal"])
    if not recycled_any:
        recs.append({
            "title": "Start recycling",
            "detail": "Separate recyclables to reduce lifecycle emissions from new material production.",
            "estimate_kg_per_month": 5,
            "why": "Recycling reduces material production emissions."
        })
    else:
        recs.append({
            "title": "Good recycling habits",
            "detail": "Consider composting food waste where available.",
            "estimate_kg_per_month": 3,
            "why": "Composting avoids landfill methane."
        })

    if get("Diet_omnivore", 0) == 1:
        recs.append({
            "title": "Eat more plant-based meals",
            "detail": "Reducing red meat/dairy intake cuts diet-related emissions.",
            "estimate_kg_per_month": 25,
            "why": "Animal products can be high-carbon per serving."
        })

    recs_sorted = sorted(recs, key=lambda x: x.get("estimate_kg_per_month", 0), reverse=True)
    summary = {
        "predicted_monthly_kg": int(prediction),
        "top_tip": recs_sorted[0]["title"] if recs_sorted else "No suggestions",
        "total_potential_saving": round(sum(r.get("estimate_kg_per_month", 0) for r in recs_sorted),1)
    }
    return {"summary": summary, "recommendations": recs_sorted}
