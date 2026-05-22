"""
Generates synthetic spatiotemporal conflict dataset mimicking ACLED + NDVI + CHIRPS
data for Nigeria's 774 LGAs across 14 years (2010-2024).
Captures known conflict drivers: NDVI decline, rainfall deficit, seasonal peaks,
spatial spillover, and economic stress.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

CONFLICT_STATES = {
    "Plateau":   {"base_risk": 0.35, "seasonal_amp": 0.25, "lgas": 17},
    "Benue":     {"base_risk": 0.32, "seasonal_amp": 0.22, "lgas": 23},
    "Kaduna":    {"base_risk": 0.28, "seasonal_amp": 0.20, "lgas": 23},
    "Taraba":    {"base_risk": 0.30, "seasonal_amp": 0.24, "lgas": 16},
    "Nasarawa":  {"base_risk": 0.26, "seasonal_amp": 0.18, "lgas": 13},
    "Niger":     {"base_risk": 0.20, "seasonal_amp": 0.15, "lgas": 25},
    "Kogi":      {"base_risk": 0.18, "seasonal_amp": 0.14, "lgas": 21},
    "Adamawa":   {"base_risk": 0.22, "seasonal_amp": 0.18, "lgas": 21},
    "Zamfara":   {"base_risk": 0.25, "seasonal_amp": 0.20, "lgas": 14},
    "Kebbi":     {"base_risk": 0.15, "seasonal_amp": 0.12, "lgas": 21},
    "Lagos":     {"base_risk": 0.04, "seasonal_amp": 0.03, "lgas": 20},
    "Oyo":       {"base_risk": 0.08, "seasonal_amp": 0.06, "lgas": 33},
}

PLANTING_MONTHS = [4, 5, 6, 7]
DRY_SEASON_MONTHS = [11, 12, 1, 2, 3]
NDVI_SEASONALITY = {
    1:0.62, 2:0.58, 3:0.65, 4:0.78, 5:0.88, 6:0.91,
    7:0.89, 8:0.85, 9:0.80, 10:0.72, 11:0.65, 12:0.60
}


def generate_lga_grid() -> pd.DataFrame:
    lgas = []
    lga_id = 1
    for state, props in CONFLICT_STATES.items():
        for i in range(props["lgas"]):
            lat_base = {"Plateau": 9.2, "Benue": 7.3, "Kaduna": 10.5,
                        "Taraba": 8.9, "Nasarawa": 8.5, "Niger": 10.0,
                        "Kogi": 7.8, "Adamawa": 9.3, "Zamfara": 12.0,
                        "Kebbi": 11.5, "Lagos": 6.5, "Oyo": 7.8}.get(state, 9.0)
            lon_base = {"Plateau": 9.0, "Benue": 8.5, "Kaduna": 7.7,
                        "Taraba": 11.5, "Nasarawa": 8.3, "Niger": 5.5,
                        "Kogi": 6.7, "Adamawa": 13.0, "Zamfara": 6.2,
                        "Kebbi": 4.2, "Lagos": 3.3, "Oyo": 3.9}.get(state, 8.0)
            lgas.append({
                "lga_id": f"NG-{state[:3].upper()}-{lga_id:03d}",
                "lga_name": f"{state} LGA {i+1:02d}",
                "state": state,
                "latitude": lat_base + np.random.normal(0, 0.5),
                "longitude": lon_base + np.random.normal(0, 0.5),
                "base_conflict_risk": props["base_risk"] + np.random.uniform(-0.05, 0.05),
                "seasonal_amplitude": props["seasonal_amp"],
                "cattle_route_proximity": np.random.choice([0, 1], p=[0.55, 0.45]),
                "border_lga": np.random.choice([0, 1], p=[0.75, 0.25]),
                "poverty_index": np.random.uniform(0.45, 0.85),
                "population_density": np.random.uniform(50, 800),
            })
            lga_id += 1
    return pd.DataFrame(lgas)


def generate_monthly_features(lgas: pd.DataFrame, start: str = "2010-01", months: int = 168) -> pd.DataFrame:
    date_range = pd.period_range(start=start, periods=months, freq="M")
    records = []

    for _, lga in lgas.iterrows():
        prev_incidents = 0
        ndvi_baseline = NDVI_SEASONALITY.copy()

        for period in date_range:
            month = period.month
            year = period.year

            ndvi_base = ndvi_baseline[month]
            la_nina_year = year in [2011, 2016, 2021]
            rainfall_anomaly = np.random.normal(-0.15 if la_nina_year else 0, 0.2)
            ndvi_anomaly = np.random.normal(-0.08 if la_nina_year else 0, 0.12)
            ndvi_current = np.clip(ndvi_base + ndvi_anomaly, 0.2, 1.0)

            is_planting = 1 if month in PLANTING_MONTHS else 0
            is_dry = 1 if month in DRY_SEASON_MONTHS else 0
            food_price_index = 100 + np.random.normal(0, 12) + (year - 2010) * 2.5
            food_price_anomaly = food_price_index / 100 - 1

            risk_score = (
                lga["base_conflict_risk"]
                + lga["seasonal_amplitude"] * is_planting * 0.8
                + lga["cattle_route_proximity"] * 0.08
                + max(0, -ndvi_anomaly) * 0.35
                + max(0, -rainfall_anomaly) * 0.25
                + max(0, food_price_anomaly) * 0.20
                + prev_incidents * 0.18
                + lga["poverty_index"] * 0.10
                + is_planting * 0.12
                + np.random.normal(0, 0.05)
            )
            risk_score = np.clip(risk_score, 0, 1)

            incident = 1 if (np.random.random() < risk_score * 0.8) else 0
            n_incidents = np.random.poisson(risk_score * 2.5) if incident else 0
            n_fatalities = np.random.poisson(n_incidents * 1.8) if n_incidents > 0 else 0

            records.append({
                "lga_id": lga["lga_id"],
                "state": lga["state"],
                "latitude": lga["latitude"],
                "longitude": lga["longitude"],
                "period": str(period),
                "year": year,
                "month": month,
                "ndvi_current": round(ndvi_current, 4),
                "ndvi_anomaly": round(ndvi_anomaly, 4),
                "rainfall_anomaly": round(rainfall_anomaly, 4),
                "food_price_index": round(food_price_index, 2),
                "food_price_anomaly": round(food_price_anomaly, 4),
                "is_planting_season": is_planting,
                "is_dry_season": is_dry,
                "cattle_route_proximity": lga["cattle_route_proximity"],
                "poverty_index": round(lga["poverty_index"], 3),
                "population_density": round(lga["population_density"], 1),
                "incidents_lag1m": prev_incidents,
                "n_conflict_incidents": n_incidents,
                "n_fatalities": n_fatalities,
                "conflict_flag": incident,
                "conflict_risk_score": round(risk_score, 4),
            })
            prev_incidents = n_incidents

    return pd.DataFrame(records)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["lga_id", "period"])
    g = df.groupby("lga_id")
    for lag in [1, 2, 3, 6, 12]:
        df[f"incidents_lag{lag}m"] = g["n_conflict_incidents"].shift(lag)
        df[f"ndvi_lag{lag}m"] = g["ndvi_current"].shift(lag)
    df["incidents_roll3m"] = g["n_conflict_incidents"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["incidents_roll6m"] = g["n_conflict_incidents"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).sum()
    )
    df["ndvi_trend_3m"] = (
        df["ndvi_current"] / df["ndvi_lag3m"].replace(0, np.nan)
    ).clip(0.5, 2.0)
    df["target_conflict_90d"] = g["conflict_flag"].transform(
        lambda x: x.shift(-3).rolling(3, min_periods=1).max()
    ).fillna(0).astype(int)
    return df


if __name__ == "__main__":
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating LGA grid...")
    lgas = generate_lga_grid()
    lgas.to_csv(out / "lga_grid.csv", index=False)

    print("Generating 14-year monthly feature matrix...")
    df = generate_monthly_features(lgas, months=168)
    df = add_lag_features(df)
    df.dropna(subset=["incidents_lag6m"]).to_csv(out / "conflict_features.csv", index=False)

    pos_rate = df["conflict_flag"].mean()
    print(f"  {len(df):,} records | {len(lgas)} LGAs | conflict rate: {pos_rate:.1%}")
    print(f"  Total incidents: {df['n_conflict_incidents'].sum():,}")
    print(f"  Total fatalities: {df['n_fatalities'].sum():,}")
    print("Saved to data/processed/")
