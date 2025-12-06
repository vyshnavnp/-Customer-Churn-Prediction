import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import preprocess
import os

# --- Configuration ---
DATA_PATH = "data/Customer_data.csv"  # Ensure this path is correct relative to where you run the script
EXPERIMENT_NAME = "Telco_Churn_Prediction"
MODEL_NAME = "catboost_churn_model"
ARTIFACT_PATH = "model"

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
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable auto-logging (Optional, but captures a lot of metadata automatically)
    # mlflow.catboost.autolog() 

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
        
        # Log params manually
        mlflow.log_params(params)

        # 5. Initialize and Train CatBoost
        # Note: We pass the categorical feature names directly to CatBoost
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

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")

        # 7. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        # 8. Save Model
        # Save locally for the API Docker image to pick up easily later
        local_model_path = "final_model.cbm"
        model.save_model(local_model_path)
        print(f"Model saved locally to {local_model_path}")

        # Log model to MLflow Artifacts
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=ARTIFACT_PATH,
            registered_model_name=MODEL_NAME
        )
        print("Model logged to MLflow.")

if __name__ == "__main__":
    train_model()