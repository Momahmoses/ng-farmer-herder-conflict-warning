"""
LightGBM farmer-herder conflict prediction classifier.
Uses spatial leave-one-region-out cross-validation to prevent
geographic data leakage between training and validation sets.
SHAP values explain every prediction with human-readable factor summaries.
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, classification_report)
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "ndvi_current", "ndvi_anomaly", "ndvi_trend_3m",
    "rainfall_anomaly",
    "food_price_anomaly",
    "incidents_lag1m", "incidents_lag2m", "incidents_lag3m", "incidents_lag6m",
    "incidents_roll3m", "incidents_roll6m",
    "is_planting_season", "is_dry_season",
    "month", "year",
    "cattle_route_proximity", "poverty_index", "population_density",
]

TARGET = "target_conflict_90d"


def load_features(processed_dir: str = "data/processed") -> pd.DataFrame:
    df = pd.read_csv(Path(processed_dir) / "conflict_features.csv")
    return df.dropna(subset=FEATURE_COLS + [TARGET])


def train(processed_dir: str = "data/processed") -> dict:
    df = load_features(processed_dir)
    states = df["state"].unique()

    print(f"Training on {len(df):,} LGA-months | {len(states)} states")
    print(f"Conflict rate (90d): {df[TARGET].mean():.1%}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    }

    fold_metrics = []
    print("\nSpatial leave-one-region-out cross-validation:")

    for hold_out_state in sorted(states):
        train_df = df[df["state"] != hold_out_state]
        val_df = df[df["state"] == hold_out_state]
        if len(val_df) < 50:
            continue

        X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET]
        X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)

        fold_metrics.append({
            "held_out_state": hold_out_state,
            "auc": roc_auc_score(y_val, y_prob),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "n_val": len(val_df),
        })
        print(f"  Hold-out {hold_out_state:<12} "
              f"AUC={fold_metrics[-1]['auc']:.3f} "
              f"F1={fold_metrics[-1]['f1']:.3f} "
              f"Recall={fold_metrics[-1]['recall']:.3f}")

    print("\nTraining final model on all data...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(df[FEATURE_COLS], df[TARGET])

    joblib.dump(final_model, MODEL_DIR / "conflict_model.pkl")
    joblib.dump(FEATURE_COLS, MODEL_DIR / "feature_names.pkl")

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    sample = df[FEATURE_COLS].sample(min(2000, len(df)), random_state=42)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics if k in m]))
                   for k in ["auc", "f1", "precision", "recall"]}

    print("\n=== Spatial CV Results ===")
    for k, v in avg_metrics.items():
        print(f"  Mean {k.upper()}: {v:.3f}")

    print("\n=== Top 10 Conflict Predictors (SHAP) ===")
    for feat, val in mean_shap.head(10).items():
        print(f"  {feat:<35} {val:.4f}")

    results = {
        "model": "LightGBM — Spatial Leave-One-Region-Out CV",
        "cv_metrics": avg_metrics,
        "fold_details": fold_metrics,
        "top_predictors": mean_shap.head(10).to_dict(),
        "n_features": len(FEATURE_COLS),
        "training_samples": len(df),
        "positive_rate": float(df[TARGET].mean()),
    }
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def predict_lga_risk(features: dict) -> dict:
    model = joblib.load(MODEL_DIR / "conflict_model.pkl")
    feat_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")

    X = pd.DataFrame([features])[feat_names].fillna(0)
    prob = float(model.predict_proba(X)[0, 1])
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    top_factors = sorted(
        zip(feat_names, shap_vals[0]),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]

    risk_score = round(prob * 100, 1)
    risk_level = "HIGH" if risk_score >= 65 else "MEDIUM" if risk_score >= 35 else "LOW"

    explanation_parts = []
    for feat, val in top_factors[:3]:
        direction = "increases" if val > 0 else "decreases"
        explanation_parts.append(f"{feat.replace('_', ' ')} {direction} risk")
    explanation = ". ".join(explanation_parts).capitalize() + "."

    return {
        "conflict_probability": round(prob, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "explanation": explanation,
        "top_factors": [{"factor": f, "shap_value": round(float(v), 4)} for f, v in top_factors],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    if args.train:
        train(args.processed_dir)
