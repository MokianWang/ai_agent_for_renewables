"""Tool implementations wrapping existing codebase functionality."""

import os
import sys
import json

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_dir = os.path.join(_project_root, "src")
for _p in (_project_root, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_data as _load_data, load_config
from src.utils import TASKS


def get_system_status() -> str:
    """List available models, data, and OPF configuration."""
    config = load_config(os.path.join(_project_root, "config.json"))
    models_dir = os.path.join(_project_root, "models")
    data_dir = os.path.join(_project_root, "data", "CAISO")
    opf_dir = os.path.join(_project_root, "opf_results")

    models_available = []
    if os.path.exists(models_dir):
        models_available = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]

    data_files = []
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    opf_has_results = os.path.exists(os.path.join(opf_dir, "summary.csv"))

    return json.dumps({
        "models_available": models_available,
        "prediction_tasks": [f"{t}_{z}" for t, z in TASKS],
        "data_files": data_files,
        "opf_results_exist": opf_has_results,
        "config": {
            "data_path": config.get("data_path"),
            "timestep": config.get("timestep"),
            "history_hour": config.get("history_hour"),
        }
    }, indent=2)


# ─── Date-Specific Data & Prediction ────────────────────────────────

def _get_date_data(zone: int, date_str: str):
    """Load and normalize data for a zone filtered to target date + lookback."""
    import pandas as pd

    config = load_config(os.path.join(_project_root, "config.json"))
    history_hour = config.get("history_hour", 6)

    data = _load_data(zone, config)
    target_start = pd.Timestamp(date_str)
    lookback_start = target_start - pd.Timedelta(hours=history_hour)
    data_end = target_start + pd.Timedelta(hours=24)
    data = data[(data.index >= lookback_start) & (data.index < data_end)]

    if len(data) == 0:
        raise ValueError(f"No data found for {date_str} in zone {zone}")

    data = data.copy()
    data["Solar Zenith Angle"] /= 180.0
    data["Relative Humidity"] /= 100.0
    for col in ["DHI", "DNI", "GHI", "Dew Point", "Wind Speed", "Temperature"]:
        mn, mx = data[col].min(), data[col].max()
        if mn != 0:
            data[col] = (data[col] - mn) / (mx - mn)
        else:
            data[col] /= mx if mx != 0 else 1.0

    return data, config


def _predict_for_date(prediction_type: str, zone: int, date_str: str):
    """Run prediction for a specific date using a trained LightGBM model."""
    import pandas as pd
    import numpy as np
    import joblib
    from src.data_loader import series_to_supervised

    data, config = _get_date_data(zone, date_str)
    history_hour = config.get("history_hour", 6)
    timestep = config.get("timestep", 15)
    x = history_hour * 60 // timestep

    model_path = os.path.join(_project_root, "models", f"{prediction_type}_{zone}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {prediction_type}_{zone}.joblib")

    model = joblib.load(model_path)
    weather_cols = ["DHI", "DNI", "GHI", "Dew Point", "Solar Zenith Angle",
                    "Wind Speed", "Relative Humidity", "Temperature"]

    supervised = series_to_supervised(data[prediction_type], n_in=x, n_out=1, dropnan=True)
    power_input = supervised.iloc[:, :-1]
    weather = data[weather_cols].loc[supervised.index]

    combined = pd.concat([weather.reset_index(drop=True), power_input.reset_index(drop=True)], axis=1)
    combined.index = power_input.index

    target_start = pd.Timestamp(date_str)
    target_end = target_start + pd.Timedelta(hours=24)
    combined = combined[(combined.index >= target_start) & (combined.index < target_end)]

    if len(combined) == 0:
        raise ValueError(f"No prediction samples for {date_str} zone {zone}")

    y_pred = np.clip(model.predict(combined.values), 0, None)

    os.makedirs(os.path.join(_project_root, "predictions"), exist_ok=True)
    pd.DataFrame({"Predicted": y_pred}).to_csv(
        os.path.join(_project_root, "predictions", f"{prediction_type}_{zone}.csv"), index=False
    )

    return {
        "prediction_type": prediction_type,
        "zone": zone,
        "samples": len(y_pred),
        "mean": round(float(np.mean(y_pred)), 6),
        "max": round(float(np.max(y_pred)), 6),
        "min": round(float(np.min(y_pred)), 6),
    }


def load_data_for_date(date_str: str) -> str:
    """Load CAISO data for a specific date across all 4 zones."""
    results = {}
    for zone in [1, 2, 3, 4]:
        try:
            data, _ = _get_date_data(zone, date_str)
            results[f"zone_{zone}"] = {
                "rows": len(data),
                "range": [str(data.index.min()), str(data.index.max())],
            }
        except Exception as e:
            results[f"zone_{zone}"] = {"error": str(e)}
    return json.dumps({"date": date_str, "zones": results}, indent=2)


def run_forecast_for_date(prediction_type: str, zone: int, date_str: str) -> str:
    """Run a single prediction for a specific date."""
    try:
        result = _predict_for_date(prediction_type, zone, date_str)
        return json.dumps({**result, "status": "success"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"})


def run_full_forecast_for_date(date_str: str) -> str:
    """Run all 9 predictions for a specific date. Saves CSVs to predictions/."""
    results = []
    errors = []

    for ptype, zone in TASKS:
        try:
            results.append(_predict_for_date(ptype, zone, date_str))
        except Exception as e:
            errors.append(f"{ptype}_{zone}: {e}")

    return json.dumps({
        "date": date_str,
        "completed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }, indent=2)


# ─── OPF Dispatch ───────────────────────────────────────────────────

def run_opf() -> str:
    """Run forecasts then execute AC OPF dispatch for 96 time steps."""
    from src.dispatch import run_opf_dispatch

    forecast_result = run_full_forecast_for_date("2020-06-01")
    forecast_data = json.loads(forecast_result)
    if forecast_data.get("failed", 0) > 0:
        return json.dumps({"error": f"{forecast_data['failed']} predictions failed", "status": "failed"})

    results, pg, costs = run_opf_dispatch()
    return json.dumps({
        "status": "success",
        "time_steps": len(results),
        "convergence_rate": round(sum(1 for r in results if r["success"]) / len(results) * 100, 1),
        "average_cost": round(float(sum(r["cost"] for r in results) / len(results)), 2),
        "total_cost": round(float(sum(r["cost"] for r in results)), 2),
        "results_dir": os.path.join(_project_root, "opf_results"),
    })


def run_dispatch_for_date(date_str: str) -> str:
    """End-to-end dispatch for a specific date: load data → forecast → OPF."""
    forecast_result = run_full_forecast_for_date(date_str)
    forecast_data = json.loads(forecast_result)

    if forecast_data["failed"] > 0:
        return json.dumps({
            "date": date_str,
            "status": "partial",
            "forecast": forecast_data,
            "error": f"{forecast_data['failed']} predictions failed",
        })

    from src.dispatch import run_opf_dispatch
    results, pg, costs = run_opf_dispatch()

    return json.dumps({
        "date": date_str,
        "status": "success",
        "forecast": {"completed": forecast_data["completed"], "results": forecast_data["results"]},
        "opf": {
            "status": "success",
            "time_steps": len(results),
            "convergence_rate": round(sum(1 for r in results if r["success"]) / len(results) * 100, 1),
            "average_cost": round(float(sum(r["cost"] for r in results) / len(results)), 2),
            "total_cost": round(float(sum(r["cost"] for r in results)), 2),
        },
    }, indent=2)


# ─── Analysis ───────────────────────────────────────────────────────

def analyze_opf_results() -> str:
    """Analyze OPF results: costs, convergence, renewable vs conventional generation."""
    import pandas as pd
    import numpy as np

    results_dir = os.path.join(_project_root, "opf_results")
    summary_path = os.path.join(results_dir, "summary.csv")

    if not os.path.exists(summary_path):
        return json.dumps({"error": "No OPF results found. Run dispatch first."})

    summary = pd.read_csv(summary_path)
    costs = summary["total_cost"].values
    converged = summary["converged"].values

    gen_breakdown = {}
    for t in [1, 10, 30, 50, 70, 90]:
        gen_file = os.path.join(results_dir, f"t_{t}_gen.csv")
        if os.path.exists(gen_file):
            gen = pd.read_csv(gen_file, index_col=0)
            total_p = gen["p_mw"].sum()
            renewable_p = gen[gen["p_mw"] > 0.1]["p_mw"].sum()
            conventional_p = gen[gen["p_mw"] <= 0.1]["p_mw"].sum()
            gen_breakdown[f"t_{t}"] = {
                "total_mw": round(float(total_p), 4),
                "renewable_mw": round(float(renewable_p), 4),
                "conventional_mw": round(float(conventional_p), 4),
                "renewable_pct": round(float(renewable_p / total_p * 100 if total_p > 0 else 0), 1),
            }

    return json.dumps({
        "time_steps": len(summary),
        "convergence_rate": round(float(np.mean(converged) * 100), 1),
        "average_cost": round(float(np.mean(costs)), 2),
        "min_cost": round(float(np.min(costs)), 2),
        "max_cost": round(float(np.max(costs)), 2),
        "total_cost": round(float(np.sum(costs)), 2),
        "generation_breakdown": gen_breakdown,
    }, indent=2)


# Tool registry
TOOLS = {
    "get_system_status": get_system_status,
    "load_data_for_date": load_data_for_date,
    "run_forecast_for_date": run_forecast_for_date,
    "run_full_forecast_for_date": run_full_forecast_for_date,
    "run_opf": run_opf,
    "run_dispatch_for_date": run_dispatch_for_date,
    "analyze_opf_results": analyze_opf_results,
}
