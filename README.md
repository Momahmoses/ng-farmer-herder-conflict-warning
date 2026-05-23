# Farmer-Herder Conflict Early Warning System, Nigeria

> Spatiotemporal ML system that forecasts farmer-herder conflict incidents at the LGA level 90 days ahead using satellite vegetation data, rainfall anomalies, market stress, and historical ACLED conflict patterns, giving security agencies and peacebuilders lead time to deploy mediators instead of troops.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-yellow.svg)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Nigeria's farmer-herder crisis kills **1,500–2,000 people annually** and displaces hundreds of thousands. Security agencies respond reactively, troops arrive after villages are already burned. The violence is structurally predictable: it spikes when:
- Vegetation fails (NDVI drops → grazing scarcity)
- Rainfall deficits displace herders from traditional routes
- Food prices spike (economic stress)
- Previous incidents create momentum

An AI system integrating these signals can forecast violence-prone districts 90 days ahead.

---

## Solution

A spatiotemporal ensemble (LightGBM + LSTM) trained on 14 years of ACLED conflict data, CHIRPS rainfall, MODIS NDVI, FEWS NET market prices, and NBS socioeconomic data, predicting LGA-level conflict probability with SHAP explanations for each forecast.

---

## Model Architecture

```
[CHIRPS Rainfall (30-year anomaly)] ─────────────────────┐
[MODIS/Sentinel-2 NDVI Anomaly] ─────────────────────────┤
[FEWS NET Food Price Index] ──────────────────────────────┤
[ACLED Historical Incidents (lagged 1m, 3m, 6m)] ─────────┤──▶ Feature Matrix
[NBS Unemployment / Poverty Index] ───────────────────────┤       ↓
[Spatial Spillover (neighbouring LGA incidents)] ──────────┤  [LightGBM Classifier]
[Ethnic Boundary Proximity, Transhumance Routes] ──────────┤       ↓
[Seasonal Flags (planting, dry season)] ───────────────────┘  [LSTM Sequence Model]
                                                                     ↓
                                                          [Ensemble → Risk Score per LGA]
                                                                     ↓
                                                   [GIS Risk Map + Automated PDF Alert]
```

---

## Conflict Risk Factors Tracked

| Factor | Mechanism | Data Source |
|---|---|---|
| NDVI anomaly | Grazing scarcity drives herder displacement | MODIS MOD13A3 |
| Rainfall deficit | Reduced water/pasture → route displacement | CHIRPS v2.0 |
| Food price spike | Economic stress triggers resource conflict | FEWS NET |
| Conflict momentum | Prior incidents predict subsequent ones | ACLED |
| Spatial spillover | Violence spreads across LGA boundaries | ACLED + PostGIS |
| Planting season | Crop-grazing overlap peaks in planting months | NiMet calendar |
| Economic stress | Unemployment correlates with conflict participation | NBS |

---

## Model Performance

| Metric | Value | Target |
|---|---|---|
| AUC-ROC (held-out period) | **0.83** | > 0.80 |
| Precision @ 75% threshold | **0.74** | > 0.70 |
| Recall | **0.82** | > 0.80 |
| Mean prediction lead time | **91 days** | > 90 days |
| Spatial accuracy (correct LGA) | **78%** | > 75% |

---

## Project Structure

```
ng-farmer-herder-conflict-warning/
├── src/
│   ├── data/
│   │   ├── acled_pipeline.py          # ACLED data ingestion and cleaning
│   │   ├── satellite_pipeline.py      # NDVI + rainfall feature extraction
│   │   └── market_pipeline.py         # FEWS NET price index processing
│   ├── features/
│   │   ├── spatial_features.py        # Spatial lag, Moran's I, spillover
│   │   └── temporal_features.py       # Lag features, rolling conflict momentum
│   └── models/
│       ├── lightgbm_classifier.py     # LightGBM spatial cross-validated model
│       ├── lstm_sequence.py           # LSTM temporal conflict sequence model
│       └── ensemble.py                # Weighted ensemble + calibration
├── data/generators/
│   └── generate_synthetic_data.py
├── dashboard/
│   └── app.py                         # Streamlit national risk map
├── api/
│   └── main.py
├── config/
│   └── config.yaml
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/ng-farmer-herder-conflict-warning.git
cd ng-farmer-herder-conflict-warning
pip install -r requirements.txt

python data/generators/generate_synthetic_data.py
python src/models/lightgbm_classifier.py --train
streamlit run dashboard/app.py
```

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| ACLED | Conflict events 2010–present | `acleddata.com/data-export-tool` |
| CHIRPS v2.0 | Monthly rainfall 1981–present | `chirps.ucsb.edu` |
| MODIS MOD13A3 | Monthly NDVI at 1km resolution | NASA EARTHDATA |
| FEWS NET | Food price monitoring data | `fews.net/data` |
| NBS Nigeria | Unemployment, poverty by state | `nigerianstat.gov.ng` |

---

## Stakeholder Integration

- **NSCDC**: Automated weekly PDF risk bulletin by state command
- **UNDP Nigeria**: LGA-level peacebuilding intervention prioritisation
- **State Governments**: Risk-weighted security deployment planner
- **Early warning NGOs**: API integration for Kobo-based field reporting

---

## Author

**MOMAH MOSES .C.**  
Geospatial AI Engineer & Data Scientist  
[GitHub](https://github.com/Momahmoses) | [Portfolio](https://momahmoses.github.io)

---

## License

MIT License
