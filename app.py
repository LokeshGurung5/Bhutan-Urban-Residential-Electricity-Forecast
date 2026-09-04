"""
app.py
------
Streamlit dashboard for urban residential electricity consumption
forecasting across all 19 dzongkhags in the dataset.

Run with:
    streamlit run app.py

Requires (pip install):
    streamlit pandas numpy matplotlib statsmodels scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pipeline import (
    load_raw,
    list_dzongkhags,
    run_pipeline_for_all,
    build_summary_table,
    get_seasonal_decomposition,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
)

st.set_page_config(
    page_title="Bhutan Electricity Forecast",
    page_icon="\u26a1",
    layout="wide",
)

DEFAULT_FILE = "electricity-consumption-monthly_v1.csv"


# ------------------------------------------------------------------------------
# Data loading (cached so the whole pipeline runs once per uploaded file)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes_or_path):
    return load_raw(file_bytes_or_path)


@st.cache_data(show_spinner=False)
def run_full_pipeline(_df, dzongkhags_key):
    """dzongkhags_key exists only so Streamlit's cache invalidates correctly
    when the dzongkhag list changes; the actual list is recovered from df."""
    dzongkhags = list_dzongkhags(_df)
    results = run_pipeline_for_all(_df, dzongkhags=dzongkhags)
    summary = build_summary_table(results)
    return results, summary


# ------------------------------------------------------------------------------
# Sidebar: data source
# ------------------------------------------------------------------------------
st.sidebar.title("\u26a1 Data source")
uploaded = st.sidebar.file_uploader("Upload consumption CSV/Excel", type=["csv", "xlsx", "xls"])

try:
    if uploaded is not None:
        df = load_data(uploaded)
        source_label = uploaded.name
    else:
        df = load_data(DEFAULT_FILE)
        source_label = DEFAULT_FILE
except FileNotFoundError:
    st.sidebar.warning(f"'{DEFAULT_FILE}' not found next to app.py — upload a file to continue.")
    st.stop()
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.caption(f"Loaded: **{source_label}**")

with st.sidebar:
    st.markdown("---")
    run_button = st.button("Run / Refresh forecast pipeline", type="primary", use_container_width=True)

if "results" not in st.session_state or run_button:
    with st.spinner("Fitting Seasonal Naive, Holt-Winters and SARIMA for all 19 dzongkhags... this takes a minute."):
        results, summary = run_full_pipeline(df, tuple(list_dzongkhags(df)))
        st.session_state["results"] = results
        st.session_state["summary"] = summary

results = st.session_state["results"]
summary = st.session_state["summary"]
dzongkhags = sorted(results.keys())

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.title("Bhutan Urban Residential Electricity Forecast")
st.caption(
    "Monthly urban residential consumption (GWh), forecast for 2026, "
    "trained on Jan 2015-Dec 2024 and validated against real Jan-Dec 2025 values."
)

# ------------------------------------------------------------------------------
# Top-level tabs
# ------------------------------------------------------------------------------
tab_overview, tab_detail, tab_data = st.tabs(["\U0001F30D National Overview", "\U0001F3D9\uFE0F Dzongkhag Detail", "\U0001F4C4 Data & Methodology"])

# ==============================================================================
# TAB 1: NATIONAL OVERVIEW
# ==============================================================================
with tab_overview:
    total_2026 = summary["2026 Total Forecast (GWh)"].sum()
    total_2025 = summary["2025 Actual Total (GWh)"].sum()
    pct_change = (total_2026 - total_2025) / total_2025 * 100
    avg_mape = summary["Test MAPE (%)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dzongkhags forecast", len(summary))
    c2.metric("2026 total forecast (GWh)", f"{total_2026:,.1f}")
    c3.metric("vs. 2025 actual", f"{pct_change:+.1f}%")
    c4.metric("Avg. test MAPE", f"{avg_mape:.1f}%")

    st.markdown("#### Best model & 2026 forecast by dzongkhag")
    st.dataframe(
        summary.style.format({
            "Test MAPE (%)": "{:.2f}",
            "2026 Total Forecast (GWh)": "{:.2f}",
            "2025 Actual Total (GWh)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 2026 total forecast by dzongkhag (GWh)")
        chart_df = summary.set_index("Dzongkhag")["2026 Total Forecast (GWh)"].sort_values()
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.barh(chart_df.index, chart_df.values, color="#2f5496")
        ax.set_xlabel("GWh")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Winning model distribution")
        model_counts = summary["Best Model"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(model_counts.values, labels=model_counts.index, autopct="%1.0f%%",
                colors=["#2f5496", "#a9c4eb", "#f4a259"])
        ax2.set_ylabel("")
        st.pyplot(fig2, use_container_width=True)

        st.markdown("#### Model accuracy across all dzongkhags (test MAPE)")
        st.bar_chart(summary.set_index("Dzongkhag")["Test MAPE (%)"])

    st.download_button(
        "\u2b07\uFE0F Download full summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="all_dzongkhags_2026_forecast_summary.csv",
        mime="text/csv",
    )

# ==============================================================================
# TAB 2: DZONGKHAG DETAIL
# ==============================================================================
with tab_detail:
    selected = st.selectbox("Choose a dzongkhag", dzongkhags, index=dzongkhags.index("THIMPHU") if "THIMPHU" in dzongkhags else 0)
    r = results[selected]
    series = r["series"]
    eval_df = r["eval"]
    best_model = r["best_model"]
    forecast_df = r["forecast"]
    test_actual = r["test_actual"]
    test_preds = r["test_preds"]

    st.markdown(f"### {selected.title()}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", best_model)
    m2.metric("Test MAPE", f"{eval_df.loc[best_model, 'MAPE (%)']:.2f}%")
    m3.metric("2026 total forecast", f"{forecast_df['Forecast_GWh'].sum():.2f} GWh")

    st.markdown("#### Historical consumption + 2026 forecast")
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.plot(series.index, series.values, label="Historical (2015-2025)", color="#1f77b4")
    ax3.plot(forecast_df.index, forecast_df["Forecast_GWh"], label="2026 Forecast", color="#d62728", linestyle="--")
    ax3.fill_between(forecast_df.index, forecast_df["CI_Lower"], forecast_df["CI_Upper"],
                      color="#d62728", alpha=0.15, label="Forecast interval")
    ax3.set_ylabel("Consumption (GWh)")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig3, use_container_width=True)

    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown("#### Model comparison on 2025 test period")
        st.dataframe(eval_df.style.format("{:.3f}").highlight_min(subset=["MAPE (%)"], color="#c6efce"),
                     use_container_width=True)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(test_actual.index, test_actual.values, label="Actual 2025", color="black", linewidth=2)
        for name, pred in test_preds.items():
            ax4.plot(pred.index, pred.values, linestyle="--", label=name)
        ax4.legend(fontsize=8)
        ax4.set_ylabel("GWh")
        ax4.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig4, use_container_width=True)

    with col_d:
        st.markdown("#### 2026 monthly forecast table")
        st.dataframe(forecast_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### Seasonal decomposition")
        decomp = get_seasonal_decomposition(series)
        fig5, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(decomp.trend, color="#2f5496"); axes[0].set_title("Trend", fontsize=9)
        axes[1].plot(decomp.seasonal, color="#f4a259"); axes[1].set_title("Seasonal", fontsize=9)
        axes[2].plot(decomp.resid, color="#888888"); axes[2].set_title("Residual", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)

    st.download_button(
        f"\u2b07\uFE0F Download {selected.title()} 2026 forecast (CSV)",
        forecast_df.to_csv().encode("utf-8"),
        file_name=f"{selected.lower()}_2026_forecast.csv",
        mime="text/csv",
    )

    with st.expander("Data cleaning report for this dzongkhag"):
        st.json(r["clean_report"])

# ==============================================================================
# TAB 3: DATA & METHODOLOGY
# ==============================================================================
with tab_data:
    st.markdown(f"""
#### Pipeline summary

- **Target variable:** urban residential electricity consumption (GWh), per dzongkhag, per month.
- **Scope:** every dzongkhag present in the uploaded file is cleaned and forecast independently — a region's series is never mixed with another's.
- **Preprocessing:** dates parsed and standardized to month-start; embedded/duplicate header rows dropped; target coerced to numeric; duplicate months de-duplicated; each dzongkhag's own missing months are linearly interpolated (not filled from another region's data).
- **Train period:** {TRAIN_START} to {TRAIN_END} (120 months).
- **Test period:** {TEST_START} to {TEST_END} (12 months, real values withheld from training to check accuracy).
- **Models compared per dzongkhag:** Seasonal Naive (baseline), Holt-Winters (additive trend + seasonality), SARIMA(1,1,1)(1,1,1)\u2081\u2082.
- **Model selection:** lowest MAPE on the 2025 test period wins, per dzongkhag — different regions may pick different models.
- **Final forecast:** the winning model is refit on the *full* 2015-2025 series (not just the training slice) before forecasting 2026, so it uses every real observation available.

#### Known simplifications (worth noting in a report)
- SARIMA's order (1,1,1)(1,1,1,12) is fixed for every dzongkhag rather than tuned per region (e.g. via `auto_arima`); a region with a very different pattern may not be well served by this order.
- When Holt-Winters or Seasonal Naive wins, the shown confidence interval is an approximate \u00b1% band, not a statistically derived prediction interval (SARIMA's interval, when it wins, is a real one from the model's forecast variance).
- A single train/test split (one 12-month holdout) is used for model selection rather than rolling-origin cross-validation.
""")

    st.markdown("#### Raw data preview")
    st.dataframe(df.head(20), use_container_width=True)
