import numpy as np
import pandas as pd

def validate_input_data(data, required_features):
    """
    Validasi input data
    """
    missing_cols = set(required_features) - set(data.columns)
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check data types
    numeric_cols = ['active_power_mw', 'apparent_temperature_c', 'cloud_cover_percent']
    for col in numeric_cols:
        if col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")
    
    return True

def prepare_sequence(data, scaler, time_steps=24):
    """
    Prepare data sequence for LSTM
    """
    if len(data) < time_steps:
        raise ValueError(f"Need at least {time_steps} rows of data")
    
    # Scale data
    scaled = scaler.transform(data)
    
    # Get last sequence
    sequence = scaled[-time_steps:]
    
    # Reshape for LSTM
    sequence = sequence.reshape(1, time_steps, -1)
    
    return sequence