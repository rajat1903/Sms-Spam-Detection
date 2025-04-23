"""
Feature extraction module for SMS Spam Detection
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the text messages.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix and target vector
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform the text data
    X = vectorizer.fit_transform(df['message'])
    
    # Get the target variable
    y = df['label'].values
    
    return X, y

def preprocess_text(text: str) -> str:
    """
    Preprocess a single text message.
    
    Args:
        text (str): Input text message
        
    Returns:
        str: Preprocessed text
    """
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens) 