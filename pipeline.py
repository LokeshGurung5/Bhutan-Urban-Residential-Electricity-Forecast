"""
pipeline.py
-----------
Reusable forecasting pipeline for urban residential electricity consumption,
generalized to run independently across every dzongkhag in the dataset
(not just Thimphu).

Key design point: each dzongkhag's series is cleaned and gap-filled on its
OWN timeline before any modeling happens. This avoids the bug in the earlier
national-aggregate version, where a region missing several months (e.g.
Sarpang, Jan 2022-Dec 2024) would silently understate a summed total instead
of showing up as a gap to interpolate.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_COL = "consumption_urban_residents_gwh"
TRAIN_START, TRAIN_END = "2015-01-01", "2024-12-01"
TEST_START, TEST_END = "2025-01-01", "2025-12-01"
FORECAST_START, FORECAST_END = "2026-01-01", "2026-12-01"
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)


# ==============================================================================
# INGESTION
# ==============================================================================
def load_raw(file_path_or_buffer):
    """Load the worksheet (csv or excel) and normalize column names."""
    if hasattr(file_path_or_buffer, "name"):
        name = file_path_or_buffer.name
    else:
        name = str(file_path_or_buffer)

    if name.lower().endswith(".csv"):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_excel(file_path_or_buffer)

    df.columns = df.columns.str.strip().str.lower()
    if "dzongkhag" not in df.columns:
        raise ValueError("Expected a 'dzongkhag' column in the source file.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected a '{TARGET_COL}' column in the source file.")

    df["dzongkhag"] = df["dzongkhag"].astype(str).str.strip().str.upper()
    return df


def list_dzongkhags(df):
    return sorted(df["dzongkhag"].unique())


# ==============================================================================
# PREPROCESSING (run once PER dzongkhag, independently)
# ==============================================================================
def clean_dzongkhag_series(df, dzongkhag):
    """
    Isolate one dzongkhag and return a clean, gap-free monthly pd.Series
    indexed by month-start date, along with a report dict describing what
    was found/fixed.
    """
    sub = df.loc[df["dzongkhag"] == dzongkhag].copy()
    if sub.empty:
        raise ValueError(f"No rows found for dzongkhag='{dzongkhag}'.")

    report = {"dzongkhag": dzongkhag, "raw_rows": len(sub)}

    date_col = next((c for c in sub.columns if "date" in c or "month" in c), None)
    if date_col is None:
        raise ValueError("Could not find a date/month column in the source file.")

    sub["date"] = pd.to_datetime(sub[date_col], errors="coerce")
    report["rows_dropped_bad_dates"] = int(sub["date"].isna().sum())
    sub = sub.dropna(subset=["date"])
    sub["date"] = sub["date"].dt.to_period("M").dt.to_timestamp()

    sub[TARGET_COL] = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    report["missing_target_values"] = int(sub[TARGET_COL].isna().sum())
    report["negative_target_values"] = int((sub[TARGET_COL] < 0).sum())

    dup_months = sub["date"].duplicated().sum()
    report["duplicate_month_rows"] = int(dup_months)
    if dup_months > 0:
        sub = sub.sort_values("date").drop_duplicates(subset="date", keep="first")

    sub = sub.sort_values("date").set_index("date")[[TARGET_COL]]

    full_index = pd.date_range(sub.index.min(), sub.index.max(), freq="MS")
    missing_months = full_index.difference(sub.index)
    report["missing_months"] = [d.strftime("%Y-%m") for d in missing_months]
    report["n_missing_months"] = len(missing_months)

    sub = sub.reindex(full_index)
    sub.index.name = "date"
    sub[TARGET_COL] = sub[TARGET_COL].interpolate(method="linear").bfill().ffill()

    series = sub[TARGET_COL]
    report["final_length_months"] = len(series)
    return series, report


# ==============================================================================
# METRICS
# ==============================================================================
def compute_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # guard against divide-by-zero months
    safe_actual = actual.replace(0, np.nan)
    mape = np.nanmean(np.abs((safe_actual - pred) / safe_actual)) * 100
    return mae, rmse, mape


# ==============================================================================
# TRAIN / TEST EVALUATION (per dzongkhag)
# ==============================================================================
def evaluate_models(series):
    """
    Fit Seasonal Naive, Holt-Winters, and SARIMA on 2015-2024, forecast
    2025, and score each against the real 2025 values. Returns the
    evaluation table, each model's test-period predictions, and the
    name of the best model by MAPE.
    """
    train = series.loc[TRAIN_START:TRAIN_END]
    test = series.loc[TEST_START:TEST_END]
    preds = {}

    # Seasonal Naive
    snaive_pred = pd.Series(train.iloc[-12:].values, index=test.index)
    preds["Seasonal Naive"] = snaive_pred

    # Holt-Winters
    hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    preds["Holt-Winters"] = hw_model.forecast(12)

    # SARIMA
    sarima_model = SARIMAX(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False)
    preds["SARIMA"] = sarima_model.forecast(12)

    rows = []
    for name, pred in preds.items():
        mae, rmse, mape = compute_metrics(test, pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

    eval_df = pd.DataFrame(rows).set_index("Model")
    best_model_name = eval_df["MAPE (%)"].idxmin()

    return eval_df, preds, test, best_model_name


# ==============================================================================
# FINAL FORECAST (refit on full series, forecast forward 12 months)
# ==============================================================================
def forecast_forward(series, best_model_name):
    """Refit the winning model on the FULL series and forecast 12 months forward."""
    if best_model_name == "SARIMA":
        model = SARIMAX(series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False)
        res = model.get_forecast(steps=12)
        summary = res.summary_frame(alpha=0.05)
        fc_mean = summary["mean"]
        ci_lower = summary["mean_ci_lower"]
        ci_upper = summary["mean_ci_upper"]

    elif best_model_name == "Holt-Winters":
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
        fc_mean = model.forecast(12)
        ci_lower = fc_mean * 0.95
        ci_upper = fc_mean * 1.05

    else:  # Seasonal Naive
        last_12 = series.iloc[-12:].values
        idx = pd.date_range(FORECAST_START, FORECAST_END, freq="MS")
        fc_mean = pd.Series(last_12, index=idx)
        ci_lower = fc_mean * 0.90
        ci_upper = fc_mean * 1.10

    forecast_df = pd.DataFrame({
        "Forecast_GWh": fc_mean.values,
        "CI_Lower": ci_lower.values,
        "CI_Upper": ci_upper.values,
    }, index=pd.date_range(FORECAST_START, FORECAST_END, freq="MS"))
    forecast_df.index.name = "Month"

    return forecast_df


# ==============================================================================
# ORCHESTRATION: run the full pipeline for every dzongkhag
# ==============================================================================
def run_pipeline_for_all(df, dzongkhags=None, progress_callback=None):
    """
    Runs cleaning -> evaluation -> forecasting for every dzongkhag.
    progress_callback(i, n, name), if given, is called after each dzongkhag
    finishes (useful for a Streamlit progress bar).

    Returns a dict keyed by dzongkhag name:
        {
          "series": clean historical pd.Series,
          "clean_report": {...},
          "eval": eval_df,
          "test_preds": {model_name: pd.Series},
          "test_actual": pd.Series,
          "best_model": str,
          "forecast": forecast_df,
        }
    """
    if dzongkhags is None:
        dzongkhags = list_dzongkhags(df)

    results = {}
    for i, dz in enumerate(dzongkhags, start=1):
        series, clean_report = clean_dzongkhag_series(df, dz)
        eval_df, test_preds, test_actual, best_model = evaluate_models(series)
        forecast_df = forecast_forward(series, best_model)

        results[dz] = {
            "series": series,
            "clean_report": clean_report,
            "eval": eval_df,
            "test_preds": test_preds,
            "test_actual": test_actual,
            "best_model": best_model,
            "forecast": forecast_df,
        }

        if progress_callback is not None:
            progress_callback(i, len(dzongkhags), dz)

    return results


def build_summary_table(results):
    """One row per dzongkhag: best model, its MAPE, and total forecast 2026 GWh."""
    rows = []
    for dz, r in results.items():
        rows.append({
            "Dzongkhag": dz,
            "Best Model": r["best_model"],
            "Test MAPE (%)": r["eval"].loc[r["best_model"], "MAPE (%)"],
            "2026 Total Forecast (GWh)": r["forecast"]["Forecast_GWh"].sum(),
            "2025 Actual Total (GWh)": r["test_actual"].sum(),
        })
    return pd.DataFrame(rows).sort_values("Dzongkhag").reset_index(drop=True)


def get_seasonal_decomposition(series):
    return seasonal_decompose(series, model="additive", period=12)
