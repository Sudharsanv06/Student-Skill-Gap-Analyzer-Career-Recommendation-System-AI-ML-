"""
Text Preprocessing Module
Student Skill Gap Analyzer & Career Recommendation System

This module contains functions for cleaning and preprocessing skill text data.
"""

import re
import string


def lowercase_text(text):
    """
    Convert text to lowercase.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lowercase text
    """
    if isinstance(text, str):
        return text.lower()
    return text


def remove_special_characters(text):
    """
    Remove special characters from text, keeping only letters, numbers, and spaces.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with special characters removed
    """
    if isinstance(text, str):
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text
    return text


def remove_extra_spaces(text):
    """
    Remove extra whitespaces and normalize spacing.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized spacing
    """
    if isinstance(text, str):
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing spaces
        text = text.strip()
        return text
    return text


def preprocess_text(text):
    """
    Complete text preprocessing pipeline.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters
    3. Remove extra spaces
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Apply preprocessing steps
    text = lowercase_text(text)
    text = remove_special_characters(text)
    text = remove_extra_spaces(text)
    
    return text


def preprocess_skills_column(df, column_name='skills'):
    """
    Apply preprocessing to a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column to preprocess
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed column
    """
    df = df.copy()
    df[f'{column_name}_clean'] = df[column_name].apply(preprocess_text)
    return df


# Example usage
if __name__ == "__main__":
    # Test the preprocessing functions
    sample_text = "Python, NumPy & Pandas!!!"
    
    print("Original text:", sample_text)
    print("Lowercase:", lowercase_text(sample_text))
    print("Remove special chars:", remove_special_characters(sample_text))
    print("Remove extra spaces:", remove_extra_spaces(sample_text))
    print("Full preprocessing:", preprocess_text(sample_text))
