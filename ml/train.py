import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import preprocess
import os
import requests # Used to check connection

# --- Configuration ---
DATA_PATH = "../data/Customer_data.csv" 
EXPERIMENT_NAME = "Telco_Churn_Prediction"
MODEL_NAME = "catboost_churn_model"
ARTIFACT_PATH = "model"
TRACKING_URI = "http://44.211.239.119:5000"

def train_model():
    # 1. Prepare Data
    print("Loading and preprocessing data...")
    try:
        X, y, cat_features = preprocess.prepare_for_training(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}.")
        return

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Define Hyperparameters
    params = {
        "iterations": 500,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "Logloss",
        "verbose": 100,
        "random_seed": 42
    }

    # 4. Train Model
    print("Training CatBoost Model...")
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    # 5. Evaluation
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {acc:.4f}")
    
    # --- SAVE MODEL LOCALLY (CRITICAL FOR DOCKER BUILD) ---
    local_model_path = "final_model.cbm"
    model.save_model(local_model_path)
    print(f"✅ Model saved locally to {local_model_path}")

    # --- 6. OPTIONAL MLFLOW LOGGING ---
    print(f"Attempting to connect to MLflow at {TRACKING_URI}...")
    
    try:
        # Configure MLflow
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc", auc)
            
            mlflow.catboost.log_model(
                cb_model=model,
                artifact_path=ARTIFACT_PATH,
                registered_model_name=MODEL_NAME
            )
            print("✅ Successfully logged metrics to MLflow.")
            
    except Exception as e:
        print("\n⚠️ WARNING: Could not connect to MLflow Server.")
        print(f"Reason: {e}")
        print("Skipping MLflow logging, but continuing pipeline since local model is saved.")
        # We do NOT raise the error here, allowing the script to finish successfully.

if __name__ == "__main__":
    train_model()