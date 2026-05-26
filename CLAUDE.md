# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI-powered power system prediction and optimization: LightGBM models forecast solar/wind/load, then pandapower runs optimal power flow (OPF) dispatch on an IEEE 30-bus system. Two independent modules chained via CSV files in `predictions/`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 9 LightGBM models (3 solar + 2 wind + 4 load zones)
python src/train.py

# Evaluate all trained models (RMSE + plots)
python src/test.py

# Run OPF dispatch (96 time steps, expects predictions/ CSV files)
python src/dispatch.py

# Analyze renewable vs conventional generation from OPF results
python analyze_generation.py

# Verify load patterns across OPF time steps
python verify_load_and_generation.py
```

All scripts are run from the **project root**, not from `src/`. The `dispatch.py` script internally `os.chdir()` to the project root.

## Architecture

### Prediction pipeline (`src/train.py` → `src/test.py`)

```
CAISO CSV data (data/CAISO/CAISO_zone_{1-4}_.csv)
  → data_loader.load_data() → normalize() → prepare_features()
  → series_to_supervised() creates lagged windows from history_hour config
  → train/test split by date: before 2020-10-01 = train, after = test
  → LightGBM model (500 rounds, early stopping after 20)
  → saved to models/{type}_{zone}.joblib
```

**The 9 prediction tasks** are defined once in `src/utils.py:TASKS` — this is the single source of truth. Both `train.py` and `test.py` import it.

`config.json` contains mostly unused Transformer hyperparameters (model_dim, nhead, encoder/decoder layers). Only these keys matter: `data_path`, `timestep`, `resample`, `history_hour`, `test_size`, `random_state`.

`data_loader.prepare_features()` builds features by concatenating weather columns + lagged power values. The lookback window is `history_hour * 60 / timestep` steps.

### OPF dispatch (`src/dispatch.py`)

```
predictions/*.csv (Predicted column)
  → create_ieee30_base_system() builds islanded IEEE 30-bus network
  → For each of 96 time steps (24h × 15min):
      - Deep-copies the base network
      - Scales loads by area prediction factors (load × 0.6)
      - Adds 5 renewable generators (3 solar + 2 wind) at fixed buses
      - Renewable capacity = 15% of total active load, proportionally allocated
      - Renewables have zero marginal cost (dispatched first)
      - Runs pp.runopp()
  → Output: opf_results/summary.csv + per-step bus/gen CSV files
```

**Key design decisions in dispatch.py:**
- The system is **islanded** (no external grid). Bus 1 generator is the slack/reference bus.
- 6 conventional generators at buses 1, 2, 5, 8, 11, 13 with linear cost $10/MW.
- 5 renewable generators: solar at buses 12, 15, 8; wind at buses 10, 24.
- Load is scaled by the prediction factor × 0.6 (matching the MATLAB reference).
- Time step results are deep-copied from a base network — each step is independent.
- Uses `pandapower.runopp()` for AC optimal power flow (not DC).

### Shared utilities (`src/utils.py`)

`evaluate_model()` loads a `.joblib` model, runs inference on test data, saves results CSV and comparison plots. Both `train.py` (for post-training eval) and `test.py` (for standalone eval) call it — this is the only evaluation code path.

### Analysis scripts

- `analyze_generation.py` — reads OPF results and reports renewable vs conventional generation ratios across sample time steps.
- `verify_load_and_generation.py` — verifies that load varies across time steps (confirms predictions are driving the dispatch).

## Key paths (all relative to project root)

| Directory/File | Purpose |
|---|---|
| `data/CAISO/` | Raw CAISO zone CSVs (time-indexed weather + power) |
| `models/` | Trained `.joblib` files (9 models) |
| `predictions/` | Input CSVs for OPF (must have `Predicted` column) |
| `results/` | Test prediction outputs (Actual vs Predicted CSVs) |
| `figures/` | Prediction comparison plots (PNG) |
| `opf_results/` | OPF output (summary.csv + per-step bus/gen CSVs) |
| `config.json` | Config (only data_path/timestep/resample/history_hour used) |

## Dependencies

Core: `lightgbm`, `pandapower`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`. Optional: `numba` for pandapower acceleration.
