"""
Main script for SMS Spam Detection
"""
import os
from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import extract_features
from src.model import train_model, evaluate_model

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data('data/spam.csv')
    
    # Extract features
    print("Extracting features...")
    X, y = extract_features(data)
    
    # Train model
    print("Training model...")
    model = train_model(X, y)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X, y)

if __name__ == "__main__":
    main() 