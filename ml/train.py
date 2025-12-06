import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import preprocess
import os

# --- Configuration ---
DATA_PATH = "../data/Customer_data.csv" 
EXPERIMENT_NAME = "Telco_Churn_Prediction"
MODEL_NAME = "catboost_churn_model"
ARTIFACT_PATH = "model"

# --- CRITICAL UPDATE: Point to your EC2 IP ---
# Note: If you stop/start your EC2, this IP will change!
# Ideally, store this in GitHub Secrets as MLFLOW_TRACKING_URI
TRACKING_URI = "http://44.211.239.119:5000"

def train_model():
    # 1. Prepare Data
    print("Loading and preprocessing data...")
    try:
        X, y, cat_features = preprocess.prepare_for_training(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please check the path.")
        return

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Setup MLflow
    # This connects the script to your EC2 server
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        print("Starting training...")
        
        # 4. Define Hyperparameters
        params = {
            "iterations": 500,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "Logloss",
            "verbose": 100,
            "random_seed": 42
        }
        mlflow.log_params(params)

        # 5. Train Model
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50
        )

        # 6. Evaluation
        print("Evaluating model...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        # 7. Log Metrics to EC2
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        # 8. Save Model
        # Save locally (for the Docker API image build)
        local_model_path = "final_model.cbm"
        model.save_model(local_model_path)
        print(f"Model saved locally to {local_model_path}")

        # Log model to MLflow (for the Dashboard)
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=ARTIFACT_PATH,
            registered_model_name=MODEL_NAME
        )
        print(f"Model logged to MLflow at {TRACKING_URI}")

if __name__ == "__main__":
    train_model()