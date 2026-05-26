# AI Agent for Renewables

A ReAct-pattern AI agent for power system operation and dispatch. Combines LightGBM-based renewable energy forecasting with AC optimal power flow (OPF) optimization on an IEEE 30-bus system.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.agent.main
```

## Architecture

The project has two layers:

### AI Agent (`src/agent/`)

A ReAct (Reasoning + Acting) agent that accepts natural language commands and executes the 3-step dispatch workflow from the [operational guidelines](Guidelines_for_Renewable_Energy_Integrated_Power_System_Operation_and_Dispatch.pdf):

```
User: "run dispatch for 2019-05-01"
  → Thought → Action → Observation → ... → Final Answer
```

The agent streams its thinking process in real time and invokes tools to:

1. **Data Acquisition** — Load CAISO data for a target date (4 zones, weather + power features)
2. **Forecasting** — Run 9 LightGBM models (3 solar + 2 wind + 4 load) to predict 15-minute intervals over 24 hours
3. **Dispatch** — Execute AC OPF on an IEEE 30-bus system (96 time steps, 11 generators) to produce optimal unit schedules

**Tools** (defined in `src/agent/tools.py`, wrapping existing pipeline code):

| Tool | Description |
|---|---|
| `get_system_status` | List available models, data files, configuration |
| `load_data_for_date(date_str)` | Load CAISO data for a date |
| `run_forecast_for_date(type, zone, date_str)` | Single prediction for a date |
| `run_full_forecast_for_date(date_str)` | All 9 predictions, outputs to `predictions/` |
| `run_opf()` | Full OPF dispatch pipeline |
| `run_dispatch_for_date(date_str)` | End-to-end: data → forecast → OPF for a date |
| `analyze_opf_results()` | Costs, convergence rate, generation mix |

**LLM Backend**: Supports DeepSeek, OpenAI, and Anthropic APIs. Configure via `.env`.

### Prediction Pipeline (`src/train.py`, `src/test.py`, `src/utils.py`)

CAISO CSV data → normalize weather features → `series_to_supervised()` creates lagged power windows → train/test split by date (2020-10-01) → LightGBM model (500 rounds, early stopping) → saved to `models/{type}_{zone}.joblib`.

The 9 prediction tasks are defined in `src/utils.py:TASKS` — this is the single source of truth used by both training and agent tools.

### OPF Dispatch (`src/dispatch.py`)

Builds an islanded IEEE 30-bus network (6 conventional generators at buses 1, 2, 5, 8, 11, 13; 5 renewable generators at buses 8, 10, 12, 15, 24). For each 15-minute interval: deep-copies the base network, scales loads by forecast factors, adds renewable generators with zero marginal cost, runs `pandapower.runopp()`. Renewables are dispatched first due to zero cost.

### Data Loading (`src/data_loader.py`)

Loads CAISO zone CSVs (1-minute resolution, 2018–2020), resamples to configurable intervals (default 15 min), normalizes weather features, and builds supervised learning windows from configurable lookback hours.

## Project Structure

```
├── src/
│   ├── agent/              # AI agent (ReAct loop, LLM client, tools, prompt)
│   │   ├── main.py
│   │   ├── agent.py
│   │   ├── llm.py
│   │   ├── tools.py
│   │   └── prompts.py
│   ├── data_loader.py      # Data loading, normalization, feature engineering
│   ├── utils.py            # Shared evaluation, plotting, TASKS definition
│   ├── train.py            # LightGBM training (9 models)
│   ├── test.py             # Model evaluation
│   └── dispatch.py         # IEEE 30-bus AC OPF dispatch
├── data/CAISO/             # CAISO zone CSV data (gitignored)
├── models/                 # Trained .joblib models (gitignored)
├── predictions/            # Forecast CSVs for OPF input
├── opf_results/            # OPF output per time step
├── config.json             # Runtime configuration
├── requirements.txt        # Python dependencies
└── .env                    # API keys (gitignored)
```

## Usage

```bash
# Interactive agent mode
python -m src.agent.main

# Single query
python -m src.agent.main --query "run dispatch for 2019-05-01"
python -m src.agent.main --query "show system status"
python -m src.agent.main --query "analyze the latest OPF results"

# Train models (standalone)
python src/train.py

# Evaluate models (standalone)
python src/test.py
```

## Configuration

**`.env`** — LLM API keys:
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

**`config.json`** — Data pipeline:
```
data_path: CAISO data directory
timestep: resample interval in minutes (default 15)
resample: whether to resample raw data
history_hour: lookback window in hours for prediction features
```

## Dependencies

| Package | Purpose |
|---|---|
| lightgbm | Gradient boosting prediction models |
| pandapower | AC optimal power flow simulation |
| openai | LLM API client (DeepSeek/OpenAI compatible) |
| pandas, numpy | Data processing |
| scikit-learn | Train/test split, evaluation metrics |
| matplotlib | Prediction visualization |
| joblib | Model serialization |
| numba | pandapower acceleration (optional) |
