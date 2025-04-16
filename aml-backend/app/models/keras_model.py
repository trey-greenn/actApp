import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Dict, Tuple, Any

def preprocess_data(
    df: pd.DataFrame, 
    train_split: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Preprocess the transaction data for model training
    """
    # Separate features and target
    X = df.drop(['label', 'transaction_id'], axis=1)
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1-train_split), random_state=42, stratify=y
    )
    
    # Define numeric and categorical features
    numeric_features = ['amount']
    categorical_features = ['origin_country', 'destination_country', 'account_id']
    date_features = ['timestamp']
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Date preprocessing - extract components
    X_train = extract_date_features(X_train)
    X_test = extract_date_features(X_test)
    
    # Update the lists of features
    date_components = ['year', 'month', 'day', 'hour', 'minute', 'dayofweek']
    numeric_features = numeric_features + date_components
    
    # Drop the original date column
    X_train = X_train.drop(['timestamp'], axis=1)
    X_test = X_test.drop(['timestamp'], axis=1)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Get feature names
    num_features = numeric_features
    cat_features = []
    
    for feature in categorical_features:
        categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[
            categorical_features.index(feature)]
        for category in categories:
            cat_features.append(f"{feature}_{category}")
    
    feature_names = num_features + cat_features
    
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, feature_names

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract components from timestamp column
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    learning_rate: float = 0.001,
    epochs: int = 10,
    batch_size: int = 32,
    hidden_layers: List[int] = [64, 32]
) -> Tuple[Dict[str, float], List[List[int]], List[int]]:
    """
    Train a Keras neural network model for AML detection
    """
    # Get the number of features
    n_features = X_train.shape[1]
    
    # Create model
    model = Sequential()
    
    # Add input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=n_features))
    model.add(Dropout(0.2))
    
    # Add hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Find indices of flagged transactions in the test set
    flagged_indices = np.where(y_pred == 1)[0]
    
    # Create metrics dictionary
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1Score": float(f1),
        "accuracy": float(accuracy)
    }
    
    return metrics, cm, flagged_indices.tolist()