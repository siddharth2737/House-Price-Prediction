# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean the data (handle missing values and correct data types)
def clean_data(df):
    # Handle missing values if any (e.g., drop rows with missing data or fill them)
    df.dropna(inplace=True)  # Option to drop rows with missing values
    
    # Ensure the correct data types (e.g., 'price', 'area' as numeric)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    
    return df

# Encode categorical columns
def encode_data(df):
    # For binary categorical columns (yes/no), use LabelEncoder
    label_encoder = LabelEncoder()
    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # For 'furnishingstatus' column, use one-hot encoding (one-hot for three categories)
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    
    return df

# Feature Scaling (Standardize numerical features)
def scale_features(df):
    scaler = StandardScaler()
    numerical_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

# Split data into features (X) and target (y)
def split_data(df):
    X = df.drop('price', axis=1)  # Features
    y = df['price']  # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
