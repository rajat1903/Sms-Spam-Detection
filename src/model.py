"""
Model training and evaluation module for SMS Spam Detection
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X: np.ndarray, y: np.ndarray) -> MultinomialNB:
    """
    Train a Naive Bayes classifier on the given data.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        
    Returns:
        MultinomialNB: Trained model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model: MultinomialNB, X: np.ndarray, y: np.ndarray) -> None:
    """
    Evaluate the model and print metrics.
    
    Args:
        model (MultinomialNB): Trained model
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show() 