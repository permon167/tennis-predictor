import pandas as pd
import numpy as np

def load_data(file_path):
    """Load raw data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the raw data by handling missing values and duplicates."""
    data = data.drop_duplicates()
    data = data.fillna(method='ffill')  # Forward fill for missing values
    return data

def preprocess_data(file_path):
    """Load and preprocess the data."""
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    return cleaned_data

def transform_data(data):
    """Transform the data into a format suitable for analysis."""
    # Example transformation: converting categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    return data

def save_processed_data(data, output_path):
    """Save the processed data to a CSV file."""
    data.to_csv(output_path, index=False)