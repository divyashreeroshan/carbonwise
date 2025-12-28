# train_and_export.py
"""
Train and export model for Flask Carbon app.

Usage:
  python train_and_export.py --data-path "/path/to/Carbon Emission.csv" --target "CarbonEmission" --output-dir "./model_artifacts" --model rf
"""

import os
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----------------------------
# Sample template used by Flask app
# ----------------------------
SAMPLE_TEMPLATE = {
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


# ----------------------------
# Helpers & Preprocessing
# ----------------------------
def split_to_list(cell):
    """Split a string or list-like cell into normalized lowercase tokens."""
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
    # remove duplicates while preserving order
    return list(dict.fromkeys(out))


def input_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess DataFrame: expand recycling/cooking, map ordinals, get_dummies.
    """
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]

    # Expand Recycling -> Do You Recyle_<Capitalized>
    if "Recycling" in data.columns:
        rec_series = data["Recycling"].apply(split_to_list)
        for item in EXPECTED_RECYCLE_ITEMS:
            colname = f"Do You Recyle_{item.capitalize()}"
            data[colname] = rec_series.apply(lambda lst: 1 if item in lst else 0)

    # Expand Cooking_With -> Cooking_with_<item>
    if "Cooking_With" in data.columns:
        cook_series = data["Cooking_With"].apply(split_to_list)
        for item in EXPECTED_COOKING_ITEMS:
            colname = f"Cooking_with_{item}"
            data[colname] = cook_series.apply(lambda lst: 1 if item in lst else 0)

    # Map Body Type and Sex if present
    if "Body Type" in data.columns:
        data["Body Type"] = data["Body Type"].map(
            {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}
        )
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"female": 0, "male": 1})

    # One-hot categorical columns
    cat_cols = []
    for col in ["Diet", "Heating Energy Source", "Transport", "Vehicle Type"]:
        if col in data.columns:
            cat_cols.append(col)
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, dtype=int)

    # ordinal mappings
    if "How Often Shower" in data.columns:
        data["How Often Shower"] = data["How Often Shower"].map(
            {"less frequently": 0, "daily": 1, "twice a day": 2, "more frequently": 3}
        )
    if "Social Activity" in data.columns:
        data["Social Activity"] = data["Social Activity"].map({"never": 0, "sometimes": 1, "often": 2})
    if "Frequency of Traveling by Air" in data.columns:
        data["Frequency of Traveling by Air"] = data["Frequency of Traveling by Air"].map(
            {"never": 0, "rarely": 1, "frequently": 2, "very frequently": 3}
        )
    if "Waste Bag Size" in data.columns:
        data["Waste Bag Size"] = data["Waste Bag Size"].map(
            {"small": 0, "medium": 1, "large": 2, "extra large": 3}
        )
    if "Energy efficiency" in data.columns:
        data["Energy efficiency"] = data["Energy efficiency"].map({"No": 0, "Sometimes": 1, "Yes": 2})

    # Fill NaN
    data = data.fillna(0)
    return data


def remove_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove object/string columns left in the dataframe to ensure only numeric columns are used for training/scaling.
    Returns a dataframe with numeric columns only.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    return numeric_df


def align_with_sample(df: pd.DataFrame, sample_template: dict) -> pd.DataFrame:
    """
    Ensure df contains all columns specified by sample_template.
    Missing columns are added with zeros. Columns ordered with sample keys first.
    """
    df = df.copy()
    for col in sample_template.keys():
        if col not in df.columns:
            df[col] = 0
    df = df.fillna(0)
    remaining = [c for c in df.columns if c not in sample_template.keys()]
    new_order = list(sample_template.keys()) + remaining
    new_order = [c for c in new_order if c in df.columns]
    return df.loc[:, new_order]


# ----------------------------
# Model training & eval
# ----------------------------
def train_model(X, y_log, model_type="rf", random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    elif model_type == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=random_state)
    else:
        raise ValueError("model_type must be 'rf' or 'mlp'")

    model.fit(X_scaled, y_log)
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test_orig):
    X_test_scaled = scaler.transform(X_test)
    preds_log = model.predict(X_test_scaled)
    preds = np.exp(preds_log)
    mae = mean_absolute_error(y_test_orig, preds)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds))  # compatible across sklearn versions
    r2 = r2_score(y_test_orig, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ----------------------------
# Main
# ----------------------------
def main(data_path, target_col, output_dir, test_size=0.2, model_type="rf", random_state=42):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {data_path}")

    df = pd.read_csv(p)
    print(f"Loaded data {data_path} with shape {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Found: {list(df.columns)[:50]}")

    # Separate target and features
    y = df[target_col].astype(float)
    X_raw = df.drop(columns=[target_col])

    # Preprocess
    X_proc = input_preprocessing(X_raw)

    # Drop leftover non-numeric columns
    X_numeric = remove_non_numeric_columns(X_proc)
    if X_numeric.shape[1] == 0:
        raise RuntimeError("No numeric columns remain after preprocessing. Check your dataset and preprocessing steps.")

    # Align with sample template (adds missing columns with zeros)
    X_aligned = align_with_sample(X_numeric, SAMPLE_TEMPLATE)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_aligned, y, test_size=test_size, random_state=random_state)

    # Train on log(y)
    y_train_log = np.log(y_train + 1e-9)
    model, scaler = train_model(X_train, y_train_log, model_type=model_type, random_state=random_state)

    # Evaluate
    metrics = evaluate_model(model, scaler, X_test, y_test)
    print("Evaluation (on holdout):", metrics)

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.sav")
    scaler_path = os.path.join(output_dir, "scale.sav")
    features_path = os.path.join(output_dir, "features.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(list(X_aligned.columns), f)

    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")
    print(f"Saved features -> {features_path}")
    print("Training & export complete.")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name in CSV (e.g. CarbonEmission)")
    parser.add_argument("--output-dir", type=str, default="./model_artifacts", help="Directory to save model & scaler")
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf", help="Model type: rf or mlp")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        target_col=args.target,
        output_dir=args.output_dir,
        test_size=args.test_size,
        model_type=args.model,
    )
