"""
Utility functions for SMS Spam Detection
"""
import os
import pickle
from typing import Any

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model (Any): Trained model
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Any: Loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_project_root() -> str:
    """
    Get the root directory of the project.
    
    Returns:
        str: Path to the project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 