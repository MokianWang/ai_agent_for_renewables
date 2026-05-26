"""Data Loading and Preprocessing"""

import pandas as pd
import numpy as np
import json
import os


def load_config(path="config.json"):
    """Load configuration."""
    with open(path, 'r') as f:
        return json.load(f)


def load_data(zone, config):
    """Load zone data."""
    path = os.path.join(config.get("data_path", "data/CAISO"), f"CAISO_zone_{zone}_.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")
    
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    
    if config.get("resample", False):
        data = data.resample(f"{config.get('timestep', 15)}min").mean()
    
    return data


def normalize(data):
    """Normalize weather features."""
    data = data.copy()
    data["Solar Zenith Angle"] /= 180.0
    data["Relative Humidity"] /= 100.0
    
    for col in ["DHI", "DNI", "GHI", "Dew Point", "Wind Speed", "Temperature"]:
        mn, mx = data[col].min(), data[col].max()
        if mn != 0:
            data[col] = (data[col] - mn) / (mx - mn)
        else:
            data[col] /= mx if mx != 0 else 1.0
    
    return data


def series_to_supervised(df, n_in=24, n_out=1, dropnan=True):
    """Convert time series to supervised learning format."""
    cols, names = [], []
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j+1}(t-{i})" for j in range(df.shape[1] if isinstance(df, pd.DataFrame) else 1)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [f"var{j+1}(t{'+' + str(i) if i > 0 else ''})" for j in range(df.shape[1] if isinstance(df, pd.DataFrame) else 1)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = df.index
    
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


def prepare_features(data, target_col, config, set_type="train"):
    """Prepare features and targets."""
    x = config.get("history_hour", 6) * 60 // config.get("timestep", 15)
    
    weather_cols = ["DHI", "DNI", "GHI", "Dew Point", "Solar Zenith Angle", 
                    "Wind Speed", "Relative Humidity", "Temperature"]
    
    # Supervised learning format
    supervised = series_to_supervised(data[target_col], n_in=x, n_out=1, dropnan=True)
    power_input = supervised.iloc[:, :-1]
    power_output = supervised.iloc[:, -1].to_frame()
    
    # Weather features
    weather = data[weather_cols].loc[supervised.index]
    
    # Combine features
    combined = pd.concat([weather.reset_index(drop=True), power_input.reset_index(drop=True)], axis=1)
    combined.index = power_input.index
    
    # Split by date
    mask = combined.index < "2020-10-01" if set_type == "train" else combined.index >= "2020-10-01"
    
    return combined[mask], power_output[mask]
