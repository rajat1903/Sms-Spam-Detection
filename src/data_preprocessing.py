"""
Data preprocessing module for SMS Spam Detection
"""
import pandas as pd
import numpy as np
from typing import Tuple

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the SMS spam dataset.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    
    # Rename columns
    df.columns = ['label', 'message']
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        test_size (float): Proportion of the dataset to include in the test split
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=42) 