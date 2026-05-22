"""
Streamlit national conflict early warning dashboard.
Displays 90-day LGA-level conflict risk scores, driving factors,
historical incident trends, and automated alert summaries.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Farmer-Herder Conflict Early Warning",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Farmer-Herder Conflict Early Warning System — Nigeria")
st.caption("90-day LGA-level conflict risk forecasting | Powered by NDVI + CHIRPS + ACLED + FEWS NET")

RISK_COLOURS = {"LOW": "#00CC00", "MEDIUM": "#FFA500", "HIGH": "#CC0000"}


@st.cache_data
def load_demo_data():
    p = Path("data/processed/conflict_features.csv")
    if p.exists():
        return pd.read_csv(p)
    from data.generators.generate_synthetic_data import generate_lga_grid, generate_monthly_features, add_lag_features
    lgas = generate_lga_grid()
    df = generate_monthly_features(lgas, months=60)
    return add_lag_features(df)


df = load_demo_data()
latest = df[df["period"] == df["period"].max()].copy()
latest["risk_score"] = (latest["conflict_risk_score"] * 100).clip(0, 100).round(1)
latest["risk_level"] = pd.cut(
    latest["risk_score"], bins=[0, 35, 65, 100], labels=["LOW", "MEDIUM", "HIGH"]
)

st.sidebar.header("Filters")
sel_state = st.sidebar.selectbox("State", ["All"] + sorted(df["state"].unique().tolist()))
threshold = st.sidebar.slider("Alert Threshold", 0, 100, 65)
show_factor = st.sidebar.selectbox(
    "Colour map by", ["risk_score", "ndvi_anomaly", "rainfall_anomaly", "incidents_roll3m"]
)

filt = latest.copy()
if sel_state != "All":
    filt = filt[filt["state"] == sel_state]

high_risk = filt[filt["risk_score"] >= threshold]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LGAs Monitored", len(filt))
with col2:
    st.metric("High-Risk LGAs", len(high_risk), delta_color="inverse")
with col3:
    st.metric("Avg NDVI Anomaly", f"{filt['ndvi_anomaly'].mean():+.3f}")
with col4:
    st.metric("Avg Rainfall Anomaly", f"{filt['rainfall_anomaly'].mean():+.3f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🗺️ Risk Map", "📈 Trend Analysis", "📋 Alert Report"])

with tab1:
    fig = px.scatter_mapbox(
        filt,
        lat="latitude", lon="longitude",
        color="risk_score",
        size="risk_score",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        hover_name="lga_name",
        hover_data={"state": True, "risk_score": True,
                    "ndvi_anomaly": ":.3f", "rainfall_anomaly": ":.3f",
                    "incidents_roll3m": True},
        mapbox_style="carto-positron",
        zoom=5, center={"lat": 9.5, "lon": 8.0},
        title=f"90-Day Conflict Risk Score by LGA — {df['period'].max()}",
        height=600,
    )
    fig.add_scattermapbox(
        lat=high_risk["latitude"],
        lon=high_risk["longitude"],
        mode="markers",
        marker=dict(size=15, color="red", opacity=0.5),
        name=f"High Risk (≥{threshold})",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        state_risk = (
            filt.groupby("state")["risk_score"]
            .agg(["mean", "max", "count"])
            .rename(columns={"mean": "avg_risk", "max": "max_risk", "count": "n_lgas"})
            .sort_values("avg_risk", ascending=False)
            .reset_index()
        )
        fig2 = px.bar(
            state_risk, x="state", y="avg_risk",
            color="avg_risk", color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            title="Average Risk Score by State",
            labels={"avg_risk": "Avg Risk Score"},
        )
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Alert threshold ({threshold})")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = px.scatter(
            filt, x="ndvi_anomaly", y="rainfall_anomaly",
            color="risk_score", size="incidents_roll3m",
            color_continuous_scale="RdYlGn_r",
            hover_name="lga_name",
            title="NDVI vs Rainfall Anomaly (bubble = recent incidents)",
            labels={"ndvi_anomaly": "NDVI Anomaly", "rainfall_anomaly": "Rainfall Anomaly"},
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig3.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Conflict Trend — Monthly (National)")
    monthly = (
        df.groupby("period")
        .agg(
            total_incidents=("n_conflict_incidents", "sum"),
            total_fatalities=("n_fatalities", "sum"),
            lgas_with_conflict=("conflict_flag", "sum"),
            avg_ndvi_anomaly=("ndvi_anomaly", "mean"),
        )
        .reset_index()
    )
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=monthly["period"], y=monthly["total_incidents"],
        name="Conflict Incidents", marker_color="red", opacity=0.7
    ))
    fig4.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["avg_ndvi_anomaly"] * 100,
        name="NDVI Anomaly (×100)", line=dict(color="green"), yaxis="y2"
    ))
    fig4.update_layout(
        title="National Monthly Conflict Incidents vs NDVI Anomaly",
        yaxis=dict(title="Incidents"),
        yaxis2=dict(title="NDVI Anomaly ×100", overlaying="y", side="right"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader(f"Active Alerts — Risk Score ≥ {threshold}")
    alerts = high_risk[["lga_name", "state", "risk_score", "risk_level",
                          "ndvi_anomaly", "rainfall_anomaly",
                          "incidents_roll3m", "cattle_route_proximity"]].copy()
    alerts = alerts.sort_values("risk_score", ascending=False)
    st.dataframe(
        alerts.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
        use_container_width=True, height=400,
    )
    st.download_button(
        "📥 Download Alert Report",
        data=alerts.to_csv(index=False),
        file_name=f"conflict_alerts_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")
