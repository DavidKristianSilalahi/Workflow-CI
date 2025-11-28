"""
Machine Learning Model Training with MLflow
Kriteria 2 - Basic Level

Author: David kristian silalahi
Date: 2025-11-25
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import json
import os
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / 'namadataset_preprocessing'
ARTIFACTS_DIR = CURRENT_DIR / 'artifacts'

# Create artifacts directory
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Setup MLflow (use file-based tracking for local)
os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_classification")


def load_preprocessing_data(data_dir: Path) -> tuple:
    """
    Load preprocessed data from directory
    
    Args:
        data_dir (Path): Directory containing preprocessed data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Loading preprocessed data...")
    
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(data_dir / 'y_test.csv').values.ravel()
    
    print(f"✓ Data loaded successfully")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame,
               y_train: np.ndarray,
               n_estimators: int = 100,
               random_state: int = 42) -> RandomForestClassifier:
    """
    Train Random Forest classifier
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (np.ndarray): Training labels
        n_estimators (int): Number of trees
        random_state (int): Random seed
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print(f"\nTraining Random Forest Classifier...")
    print(f"  Parameters: n_estimators={n_estimators}, random_state={random_state}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully")
    
    return model


def evaluate_model(model: RandomForestClassifier,
                  X_test: pd.DataFrame,
                  y_test: np.ndarray) -> dict:
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        dict: Dictionary of metrics
    """
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"✓ Model evaluated")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    
    return metrics, y_pred


def save_artifacts(model: RandomForestClassifier,
                   y_pred: np.ndarray,
                   y_test: np.ndarray,
                   metrics: dict,
                   artifacts_dir: Path) -> None:
    """
    Save model artifacts
    
    Args:
        model: Trained model
        y_pred (np.ndarray): Predictions
        y_test (np.ndarray): True labels
        metrics (dict): Metrics dictionary
        artifacts_dir (Path): Directory to save artifacts
    """
    print(f"\nSaving artifacts to {artifacts_dir}...")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(artifacts_dir / 'confusion_matrix.npy', cm)
    print(f"✓ Saved: confusion_matrix.npy")
    
    # Save classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    with open(artifacts_dir / 'classification_report.json', 'w') as f:
        json.dump(class_report, f, indent=4)
    print(f"✓ Saved: classification_report.json")
    
    # Save feature importances
    feature_importance = {
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'importance': model.feature_importances_.tolist()
    }
    with open(artifacts_dir / 'feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=4)
    print(f"✓ Saved: feature_importance.json")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("MACHINE LEARNING MODEL TRAINING - KRITERIA 2 (BASIC)")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessing_data(DATA_DIR)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        params = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': 100,
            'random_state': 42
        }
        mlflow.log_params(params)
        print("\n✓ Parameters logged to MLflow")
        
        # Train model
        model = train_model(X_train, y_train, **{k: v for k, v in params.items() if k != 'model_type'})
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        print("✓ Metrics logged to MLflow")
        
        # Save artifacts
        save_artifacts(model, y_pred, y_test, metrics, ARTIFACTS_DIR)
        
        # Log model with autolog
        mlflow.sklearn.log_model(model, "model")
        print("✓ Model logged to MLflow")
        
        # Log artifacts directory
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")
        print("✓ Artifacts logged to MLflow")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nMLflow Tracking URL: http://localhost:5000")
        print(f"Experiment: iris_classification")


if __name__ == "__main__":
    main()
